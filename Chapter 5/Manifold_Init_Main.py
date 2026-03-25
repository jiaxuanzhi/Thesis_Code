import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# ==========================================
# 0. 全局配置
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Manifold Pre-Optimizer (Genetic Algorithm) 启动... 设备: {device}")

torch.manual_seed(2024)
np.random.seed(2024)

# ==========================================
# 1. 物理环境
# ==========================================
class RadarEnvironment:
    def __init__(self, M=7, N=7, f0=10e9, array_type='Linear', R=0.5):
        self.M = M
        self.N = N
        self.f0 = f0
        self.c = 3e8
        self.lambda0 = self.c / f0
        self.d_min = self.lambda0 / 2.0
        self.array_type = array_type.capitalize()
        self.R = R
        if self.array_type == 'Linear':
            # 线性阵列边界 (根据Tex: M*lambda 和 N*lambda)
            self.chi_t = M * self.lambda0
            
            self.chi_r = N * self.lambda0
        else:
            # 平面阵列边界 (角度限制 -pi/4 到 pi/4)
            self.chi_t = np.pi / 2
            self.chi_r = np.pi / 2

# ==========================================
# 2. 遗传算法优化器 (核心逻辑)
# ==========================================
class GeneticAlgorithmOptimizer:
    def __init__(self, env, pop_size=100, mutation_rate=0.1, crossover_rate=0.8, target_sector=None):
        self.env = env
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.target_sector = target_sector # (theta, phi) for Planar integration center
        
        # 基因长度 = M (Tx) + N (Rx)
        self.gene_len = env.M + env.N
        
        # 初始化种群 (Batch, Gene_Len)
        if env.array_type == 'Linear':
            # 线性: 在 [-chi/2, chi/2] 范围内随机初始化
            # 为了更好的初始解，我们让它们尽量均匀分布然后加噪声
            range_t = torch.linspace(-env.chi_t/2, env.chi_t/2, env.M, device=device)
            range_r = torch.linspace(-env.chi_r/2, env.chi_r/2, env.N, device=device)
            base_gene = torch.cat([range_t, range_r])
            self.population = base_gene.repeat(pop_size, 1) + \
                              torch.randn(pop_size, self.gene_len, device=device) * self.env.d_min
        else:
            # 平面: 在 [-pi/2, pi/2] 范围内随机初始化
            self.population = (torch.rand(pop_size, self.gene_len, device=device) - 0.5) * np.pi

    def _sort_genes(self, pop):
        """对位置进行排序，方便计算间距约束"""
        Lt = pop[:, :self.env.M]
        Lr = pop[:, self.env.M:]
        Lt, _ = torch.sort(Lt, dim=1)
        Lr, _ = torch.sort(Lr, dim=1)
        return torch.cat([Lt, Lr], dim=1)

    def compute_fitness_linear(self, pop):
        """
        线性阵列适应度 (参考 Matlab ManifoldFitness.m)
        Obj = N * Var(Tx) + M * Var(Rx) - Penalty
        """
        # 1. 解析基因
        Lt = pop[:, :self.env.M]
        Lr = pop[:, self.env.M:]
        
        # 2. 计算方差项 (最大化)
        # Var = sum((x - mean)^2)
        mean_t = torch.mean(Lt, dim=1, keepdim=True)
        var_t = torch.sum((Lt - mean_t)**2, dim=1)
        
        mean_r = torch.mean(Lr, dim=1, keepdim=True)
        var_r = torch.sum((Lr - mean_r)**2, dim=1)
        
        # 目标函数值
        fitness = self.env.N * var_t + self.env.M * var_r
        
        # 3. 惩罚项 (Penalty)
        penalty = torch.zeros_like(fitness)
        
        # (a) 边界约束
        bound_t = self.env.chi_t / 2
        bound_r = self.env.chi_r / 2
        penalty += torch.sum(torch.relu(torch.abs(Lt) - bound_t), dim=1) * 1e6
        penalty += torch.sum(torch.relu(torch.abs(Lr) - bound_r), dim=1) * 1e6
        
        # (b) 最小间距约束
        # 假设基因已排序 (在进化循环中保证)
        diff_t = Lt[:, 1:] - Lt[:, :-1]
        diff_r = Lr[:, 1:] - Lr[:, :-1]
        
        penalty += torch.sum(torch.relu(self.env.d_min - diff_t), dim=1) * 1e7
        penalty += torch.sum(torch.relu(self.env.d_min - diff_r), dim=1) * 1e7
        
        return fitness - penalty

    def compute_fitness_planar(self, pop):
        """
        平面阵列适应度 
        Obj = Integral( sqrt( (1+|A_th|)(1+|A_ph|) - |A_th_ph|^2 ) )
        """
        Lt = pop[:, :self.env.M] # Beta_t
        Lr = pop[:, self.env.M:] # Beta_r
        
        # 1. 确定积分区域 (Batch processing for integration grid)
        # 目标区域: target +/- 3度
        if self.target_sector is None:
            th_c, ph_c = np.deg2rad(45), np.deg2rad(45)
        else:
            th_c, ph_c = self.target_sector
            
        delta = np.deg2rad(3.0) 
        # 网格点数 (少量点即可近似积分)
        K_grid = 16 
        t_grid = torch.linspace(th_c, th_c + delta, 4, device=device)
        p_grid = torch.linspace(ph_c, ph_c + delta, 4, device=device)
        T, P = torch.meshgrid(t_grid, p_grid, indexing='ij')
        theta = T.flatten() # (K,)
        phi = P.flatten()   # (K,)
        
        # 2. 坐标转换 (Batch, M)
        kR = 2 * np.pi / self.env.lambda0 * self.env.R
        
        # Xt: (Batch, 3, M)
        Xt = kR * torch.stack([torch.cos(Lt), torch.sin(Lt), torch.zeros_like(Lt)], dim=1)
        Xr = kR * torch.stack([torch.cos(Lr), torch.sin(Lr), torch.zeros_like(Lr)], dim=1)
        
        # 3. 导数向量 u (3, K)
        u_dtheta = torch.stack([
            torch.cos(theta)*torch.cos(phi), 
            torch.cos(theta)*torch.sin(phi), 
            -torch.sin(theta)
        ], dim=0)
        
        u_dphi = torch.stack([
            -torch.sin(theta)*torch.sin(phi), 
            torch.sin(theta)*torch.cos(phi), 
            torch.zeros_like(theta)
        ], dim=0)
        
        # 4. 计算矩阵项 (Batch, K)
        # 利用 Einstein Summation 快速计算
        # Xt: (B, 3, M), u: (3, K) -> Xt.T @ u -> (B, M, K)
        proj_t_dt = torch.einsum('bcm,ck->bmk', Xt, u_dtheta)
        proj_r_dt = torch.einsum('bcn,ck->bnk', Xr, u_dtheta)
        
        proj_t_dp = torch.einsum('bcm,ck->bmk', Xt, u_dphi)
        proj_r_dp = torch.einsum('bcn,ck->bnk', Xr, u_dphi)
        
        # A_theta = N * ||Xt_dt||^2 + M * ||Xr_dt||^2 + 2 * sum(Xt_dt) * sum(Xr_dt)
        # sum over elements (dim 1)
        term1_t = self.env.N * torch.sum(proj_t_dt**2, dim=1)
        term1_r = self.env.M * torch.sum(proj_r_dt**2, dim=1)
        term1_x = 2 * torch.sum(proj_t_dt, dim=1) * torch.sum(proj_r_dt, dim=1)
        Atheta = term1_t + term1_r + term1_x
        
        term2_t = self.env.N * torch.sum(proj_t_dp**2, dim=1)
        term2_r = self.env.M * torch.sum(proj_r_dp**2, dim=1)
        term2_x = 2 * torch.sum(proj_t_dp, dim=1) * torch.sum(proj_r_dp, dim=1)
        Aphi = term2_t + term2_r + term2_x
        
        # Athetaphi (Cross term)
        # N * sum(Xt_dt * Xt_dp) + ...
        cross_t = torch.sum(proj_t_dt * proj_t_dp, dim=1)
        cross_r = torch.sum(proj_r_dt * proj_r_dp, dim=1)
        
        sum_t_dt = torch.sum(proj_t_dt, dim=1)
        sum_t_dp = torch.sum(proj_t_dp, dim=1)
        sum_r_dt = torch.sum(proj_r_dt, dim=1)
        sum_r_dp = torch.sum(proj_r_dp, dim=1)
        
        Athetaphi = self.env.N * cross_t + self.env.M * cross_r + \
                    sum_t_dt * sum_r_dp + sum_t_dp * sum_r_dt # Symmetric cross
        
        # Vita = (1+|At|)(1+|Ap|) - |Atp|^2
        Vita = (1 + torch.abs(Atheta)) * (1 + torch.abs(Aphi)) - torch.abs(Athetaphi)**2
        
        # 积分: sum over K grid points
        integral = torch.sum(torch.sqrt(torch.abs(Vita) + 1e-8), dim=1)
        
        # 5. 惩罚项
        penalty = torch.zeros_like(integral)
        
        # 角度范围 -pi/2 ~ pi/2
        penalty += torch.sum(torch.relu(torch.abs(Lt) - np.pi/2), dim=1) * 1e6
        penalty += torch.sum(torch.relu(torch.abs(Lr) - np.pi/2), dim=1) * 1e6
        
        # 最小角度间隔
        d_min_angle = np.deg2rad(3.0)
        diff_t = Lt[:, 1:] - Lt[:, :-1] # 假设已排序
        diff_r = Lr[:, 1:] - Lr[:, :-1]
        penalty += torch.sum(torch.relu(d_min_angle - diff_t), dim=1) * 1e5
        penalty += torch.sum(torch.relu(d_min_angle - diff_r), dim=1) * 1e5
        
        return integral - penalty

    def evolve(self, generations=400):
        fitness_history = []
        best_gene = None
        best_fitness = -float('inf')
        
        for g in range(generations):
            # 0. 排序基因 (确保位置有序，便于计算间距)
            self.population = self._sort_genes(self.population)
            
            # 1. 计算适应度
            if self.env.array_type == 'Linear':
                fitness = self.compute_fitness_linear(self.population)
            else:
                fitness = self.compute_fitness_planar(self.population)
            
            # 记录最优
            max_fit, idx = torch.max(fitness, dim=0)
            if max_fit > best_fitness:
                best_fitness = max_fit.item()
                best_gene = self.population[idx].clone()
            
            fitness_history.append(best_fitness)
            
            # 2. 选择 (Tournament Selection)
            # 简单的概率选择：Softmax 归一化
            # 为了避免数值溢出，先减去最大值
            norm_fit = (fitness - fitness.mean()) / (fitness.std() + 1e-5)
            probs = torch.softmax(norm_fit, dim=0)
            
            # 随机选择父代索引
            parents_idx = torch.multinomial(probs, self.pop_size, replacement=True)
            parents = self.population[parents_idx]
            
            # 3. 交叉 (Arithmetic Crossover)
            # 随机配对
            parents1 = parents[:self.pop_size//2]
            parents2 = parents[self.pop_size//2:]
            
            # 生成交叉掩码
            do_cross = torch.rand(self.pop_size//2, 1, device=device) < self.crossover_rate
            alpha = torch.rand(self.pop_size//2, self.gene_len, device=device) * do_cross
            
            offspring1 = parents1 * (1 - alpha) + parents2 * alpha
            offspring2 = parents2 * (1 - alpha) + parents1 * alpha
            
            new_pop = torch.cat([offspring1, offspring2], dim=0)
            
            # 补齐种群 (如果是奇数)
            if new_pop.shape[0] < self.pop_size:
                new_pop = torch.cat([new_pop, parents[-1].unsqueeze(0)], dim=0)
            
            # 4. 变异 (Gaussian Mutation)
            mutation_mask = torch.rand_like(new_pop) < self.mutation_rate
            # 变异强度随代数衰减
            mutation_strength = 0.5 * (1 - g/generations) 
            if self.env.array_type == 'Linear':
                noise = torch.randn_like(new_pop) * self.env.d_min * mutation_strength
            else:
                noise = torch.randn_like(new_pop) * np.deg2rad(5) * mutation_strength
                
            new_pop = new_pop + noise * mutation_mask
            
            # 5. 精英保留 (Elitism)
            new_pop[0] = best_gene # 始终保留历史最优
            
            self.population = new_pop
            
            if g % 20 == 0:
                print(f"Gen {g}: Best Fitness = {best_fitness:.4f}")
        
        return best_gene, fitness_history

def run_manifold_optimization(array_type='Linear', filename='init_geom.mat', target_deg=None):
    print(f"--- 开始流形预优化 (GA): {array_type} ---")
    
    target_rad = None
    if target_deg:
        if array_type == 'Linear':
            target_rad = (np.deg2rad(target_deg[0]), 0)
        else:
            target_rad = (np.deg2rad(target_deg[0]), np.deg2rad(target_deg[1]))

    env = RadarEnvironment(array_type=array_type)
    
    # 初始化 GA
    ga = GeneticAlgorithmOptimizer(env, pop_size=100, mutation_rate=0.01, target_sector=target_rad)
    
    # 运行进化
    generations = 500 if array_type == 'Linear' else 500
    best_gene, loss_hist = ga.evolve(generations=generations)
    
    # 解析结果
    Lt_opt = best_gene[:env.M].detach().cpu().numpy()
    Lr_opt = best_gene[env.M:].detach().cpu().numpy()
    
    # 确保排序后保存
    Lt_opt.sort()
    Lr_opt.sort()
    if array_type == 'Planar':
        savemat(filename, {'L_t': Lt_opt, 'L_r': Lr_opt, 'fitness_history': loss_hist})
    else:
        savemat(filename, {'L_t': Lt_opt, 'L_r': Lr_opt, 'fitness_history': loss_hist})
    print(f"✅ GA 优化完成，结果已保存至 {filename}")
    print(f"   Tx Array: {Lt_opt}")
    print(f"   Rx Array: {Lr_opt}")
    
    # 绘图
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"GA Optimization Curve ({array_type})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 1. 线性阵列 (目标无关)
    run_manifold_optimization('Linear', 'init_linear.mat')
    
    # 2. 平面阵列 (需要指定目标区域)
    run_manifold_optimization('Planar', 'init_planar.mat', target_deg=(10, 10))