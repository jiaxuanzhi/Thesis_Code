import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.io import loadmat, savemat

# ==========================================
# 0. 全局配置
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 KID-Net (Embedded Graph Mode - HighRes Planar) 启动... 设备: {device}")

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
            self.chi_t = M * self.lambda0
            self.chi_r = N * self.lambda0
        else:
            self.chi_t = np.pi / 4
            self.chi_r = np.pi / 4

    def get_element_coords(self, L_params, is_transmit=True):
        num = self.M if is_transmit else self.N
        zeros = torch.zeros(num, device=L_params.device)
        if self.array_type == 'Linear':
            return torch.stack([L_params, zeros, zeros], dim=0)
        elif self.array_type == 'Planar':
            x = self.R * torch.cos(L_params)
            y = self.R * torch.sin(L_params)
            return torch.stack([x, y, zeros], dim=0)
        else:
            raise ValueError(f"Unknown array type")

    def steering_vector(self, positions, theta, phi):
        k = 2 * np.pi / self.lambda0
        if theta.device != positions.device: theta = theta.to(positions.device)
        if phi.device != positions.device: phi = phi.to(positions.device)
        u_vec = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=0)
        phase = k * torch.matmul(positions.T, u_vec)
        return torch.exp(1j * phase)

# ==========================================
# 2. 嵌入式网络 (Embedded Net)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(size, size)
        
    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x)))) + x

# 可选：更简洁的版本，使用Sequential
class EmbeddedUpdateNet(nn.Module):
    def __init__(self, input_dim,hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            ResidualBlock(64),
            nn.Linear(64, 128),
            nn.Tanh(),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            ResidualBlock(64),
            nn.Linear(64, input_dim)
        )
        
        # 初始化输出层
        self.net[-1].weight.data.normal_(0, 0.001)
        self.net[-1].bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. KID-Net 优化器封装
# ==========================================
class KPDNet_Optimizer(nn.Module):
    def __init__(self, env, mode='MIMO', target_angle=None, jammer_angles=None, init_file=None):
        super().__init__()
        self.env = env
        self.mode = mode 
        
        # --- 1. 阵列位置初始化 ---
        if init_file and os.path.exists(init_file):
            print(f"  📂 加载流形预优化阵列配置: {init_file}")
            data = loadmat(init_file)
            init_Lt = torch.tensor(data['L_t'].flatten(), dtype=torch.float32)
            init_Lr = torch.tensor(data['L_r'].flatten(), dtype=torch.float32)
        else:
            print("  ⚠️ 未找到初始化文件，使用默认均匀/随机分布")
            if env.array_type == 'Linear':
                init_Lt = torch.linspace(-env.chi_t/2, env.chi_t/2, env.M)
                init_Lr = torch.linspace(-env.chi_r/2, env.chi_r/2, env.N)
            else:
                init_Lt = torch.linspace(-np.pi/4, np.pi/4, env.M)
                init_Lr = torch.linspace(-np.pi/4, np.pi/4, env.N)
        
        self.L_t = nn.Parameter(init_Lt.to(device))
        self.L_r = nn.Parameter(init_Lr.to(device))
        
        self.L_t_anchor = init_Lt.to(device)
        self.L_r_anchor = init_Lr.to(device)

        # --- 2. MVDR 初始化 ---
        if target_angle is not None:
            pos_t_init = env.get_element_coords(self.L_t, True)
            pos_r_init = env.get_element_coords(self.L_r, False)
            t_theta = torch.tensor([target_angle[0]], device=device, dtype=torch.float32)
            t_phi   = torch.tensor([target_angle[1]], device=device, dtype=torch.float32)
            
            at_0 = env.steering_vector(pos_t_init, t_theta, t_phi).squeeze()
            ar_0 = env.steering_vector(pos_r_init, t_theta, t_phi).squeeze()
            
            at_j_list = []
            ar_j_list = []
            if jammer_angles:
                for j_ang in jammer_angles:
                    j_theta = torch.tensor([j_ang[0]], device=device, dtype=torch.float32)
                    j_phi   = torch.tensor([j_ang[1]], device=device, dtype=torch.float32)
                    at_j_list.append(env.steering_vector(pos_t_init, j_theta, j_phi).squeeze())
                    ar_j_list.append(env.steering_vector(pos_r_init, j_theta, j_phi).squeeze())

            w_init = self._compute_mvdr_weights(at_0, at_j_list)
            self.w_real = nn.Parameter(w_init.real)
            self.w_imag = nn.Parameter(w_init.imag)
            
            if mode == 'MIMO':
                v_0 = torch.kron(at_0, ar_0)
                v_j_list = [torch.kron(aj, arj) for aj, arj in zip(at_j_list, ar_j_list)]
                b_init = self._compute_mvdr_weights(v_0, v_j_list)
            else:
                b_init = self._compute_mvdr_weights(ar_0, ar_j_list)
            self.b_real = nn.Parameter(b_init.real)
            self.b_imag = nn.Parameter(b_init.imag)
        else:
            self.w_real = nn.Parameter(torch.randn(env.M, device=device))
            self.w_imag = nn.Parameter(torch.randn(env.M, device=device))
            dim_b = env.M * env.N if mode == 'MIMO' else env.N
            self.b_real = nn.Parameter(torch.randn(dim_b, device=device))
            self.b_imag = nn.Parameter(torch.randn(dim_b, device=device))
            
        self.dim_w = env.M
        self.dim_b = env.M * env.N if mode == 'MIMO' else env.N
        total_param_dim = (self.dim_w + self.dim_b) * 2
        
        self.net = EmbeddedUpdateNet(input_dim=total_param_dim, hidden_dim=128).to(device)
        for param in self.net.parameters():
            param.requires_grad = False

    def _compute_mvdr_weights(self, steering_vec_target, steering_vec_jammers_list):
        dim = steering_vec_target.shape[0]
        R = torch.eye(dim, device=device, dtype=torch.complex64)
        sigma_j = 100.0
        if steering_vec_jammers_list:
            for a_j in steering_vec_jammers_list:
                a_j = a_j.unsqueeze(1)
                R = R + sigma_j * torch.matmul(a_j, a_j.conj().T)
        try:
            R_inv = torch.linalg.inv(R)
        except RuntimeError:
            R = R + 0.01 * torch.eye(dim, device=device)
            R_inv = torch.linalg.inv(R)
        w_mvdr = torch.matmul(R_inv, steering_vec_target.unsqueeze(1)).squeeze()
        return w_mvdr

    def _compose_complex(self, r, i):
        c = torch.complex(r, i)
        return c / (torch.abs(c) + 1e-9)

    def forward_pass(self, target_ang, jammer_ang_list, lambda_w=0.5):
        state_vector = torch.cat([
            self.w_real, self.w_imag, 
            self.b_real, self.b_imag
        ], dim=0).unsqueeze(0) 

        delta = self.net(state_vector).squeeze()
        lr_net_scale = 0.1 
        
        d_wr = delta[0 : self.dim_w]
        d_wi = delta[self.dim_w : 2*self.dim_w]
        d_br = delta[2*self.dim_w : 2*self.dim_w + self.dim_b]
        d_bi = delta[2*self.dim_w + self.dim_b : ]

        w_real_hat = self.w_real - lr_net_scale * d_wr
        w_imag_hat = self.w_imag - lr_net_scale * d_wi
        b_real_hat = self.b_real - lr_net_scale * d_br
        b_imag_hat = self.b_imag - lr_net_scale * d_bi

        w_hat = self._compose_complex(w_real_hat, w_imag_hat)
        b_hat = self._compose_complex(b_real_hat, b_imag_hat)

        # --- 空间采样 (Space Sampling) ---
        if self.env.array_type == 'Linear':
            theta_scan = torch.linspace(-np.pi/2, np.pi/2, 360, device=device)
            phi_scan = torch.zeros_like(theta_scan)
        else:
            t = torch.linspace(-np.pi, np.pi, 360, device=device) # 俯仰角
            p = torch.linspace(-np.pi/2, np.pi/2, 180, device=device) # 方位角
            grid_t, grid_p = torch.meshgrid(t, p, indexing='ij')
            theta_scan, phi_scan = grid_t.flatten(), grid_p.flatten()

        pos_t = self.env.get_element_coords(self.L_t, True)
        pos_r = self.env.get_element_coords(self.L_r, False)
        at = self.env.steering_vector(pos_t, theta_scan, phi_scan)
        ar = self.env.steering_vector(pos_r, theta_scan, phi_scan)

        if self.mode == 'PA':
            ec = torch.abs(torch.matmul(w_hat.conj(), at))**2
        else:
            ec = torch.ones_like(theta_scan)

        if self.mode == 'MIMO':
            w_expand = w_hat.unsqueeze(1)
            at_weighted = at * w_expand
            v_virt = torch.einsum('mk,nk->mnk', at_weighted, ar).reshape(self.env.M * self.env.N, -1)
            er = torch.abs(torch.matmul(b_hat.conj(), v_virt))**2
        else:
            er = torch.abs(torch.matmul(b_hat.conj(), ar))**2

        joint_pattern = ec * er
        P_total = torch.sum(joint_pattern) + 1e-6

        # --- Loss ---
        idx_target = torch.argmin(torch.abs(theta_scan - target_ang[0]) + torch.abs(phi_scan - target_ang[1]))
        P_target = joint_pattern[idx_target]
        eta = P_target / P_total

        gamma = 0.0
        P_jammers = 0.0
        for j_ang in jammer_ang_list:
            idx_jam = torch.argmin(torch.abs(theta_scan - j_ang[0]) + torch.abs(phi_scan - j_ang[1]))
            P_jam_i = joint_pattern[idx_jam]
            gamma += P_jam_i / P_total
            P_jammers += P_jam_i

        SINR_or = P_target / (P_jammers + 1)
        sinr = 10 * torch.log10(SINR_or)
        main_loss = lambda_w * gamma - (1.0 - lambda_w) * eta

        max_val = torch.max(joint_pattern)
        constraint_pointing = torch.abs((P_target - max_val)/max_val)**2
        
        # Anchor Penalty
        dist_t = torch.norm(self.L_t - self.L_t_anchor)
        dist_r = torch.norm(self.L_r - self.L_r_anchor)
        anchor_penalty = 10.0 * (dist_t + dist_r)

        geo_penalty = 0.0
        if self.env.array_type == 'Linear':
             sorted_t, _ = torch.sort(self.L_t)
             diff_t = sorted_t[1:] - sorted_t[:-1]
             geo_penalty += torch.sum(torch.relu(self.env.d_min - diff_t))
             sorted_r, _ = torch.sort(self.L_r)
             diff_r = sorted_r[1:] - sorted_r[:-1]
             geo_penalty += torch.sum(torch.relu(self.env.d_min - diff_r))

        total_loss = main_loss + 1.0 * constraint_pointing + 0.1 * anchor_penalty + 10.0 * geo_penalty - SINR_or/10000000.0
        
        return total_loss, joint_pattern, theta_scan, phi_scan, sinr

    def update_geometry_constrained(self):
        limit = self.env.lambda0 / 10.0
        with torch.no_grad():
            self.L_t.data = torch.max(torch.min(self.L_t.data, self.L_t_anchor + limit), self.L_t_anchor - limit)
            self.L_r.data = torch.max(torch.min(self.L_r.data, self.L_r_anchor + limit), self.L_r_anchor - limit)
            
            if self.env.array_type == 'Linear':
                self.L_t.data = torch.clamp(self.L_t.data, -self.env.chi_t/2, self.env.chi_t/2)
                self.L_r.data = torch.clamp(self.L_r.data, -self.env.chi_r/2, self.env.chi_r/2)
                
                def separate(tensor, min_dist):
                    vals, _ = torch.sort(tensor)
                    for i in range(1, len(vals)):
                        if vals[i] - vals[i-1] < min_dist:
                            vals[i] = vals[i-1] + min_dist + 1e-4
                    return vals
                self.L_t.data = separate(self.L_t.data, self.env.d_min)
                self.L_r.data = separate(self.L_r.data, self.env.d_min)
            else:
                self.L_t.data = torch.clamp(self.L_t.data, -np.pi/2, np.pi/2)
                self.L_r.data = torch.clamp(self.L_r.data, -np.pi/2, np.pi/2)

# ==========================================
# 4. 辅助函数
# ==========================================
def calculate_benchmarks(model, target, jammers):
    pos_t = model.env.get_element_coords(model.L_t, True)
    pos_r = model.env.get_element_coords(model.L_r, False)
    
    t_theta = torch.tensor([target[0]], device=device, dtype=torch.float32)
    t_phi   = torch.tensor([target[1]], device=device, dtype=torch.float32)
    
    at_0 = model.env.steering_vector(pos_t, t_theta, t_phi).squeeze()
    ar_0 = model.env.steering_vector(pos_r, t_theta, t_phi).squeeze()
    
    at_j_list, ar_j_list = [], []
    for j in jammers:
        j_t = torch.tensor([j[0]], device=device, dtype=torch.float32)
        j_p = torch.tensor([j[1]], device=device, dtype=torch.float32)
        at_j_list.append(model.env.steering_vector(pos_t, j_t, j_p).squeeze())
        ar_j_list.append(model.env.steering_vector(pos_r, j_t, j_p).squeeze())

    # CBF
    w_cbf = at_0 / torch.abs(at_0)
    if model.mode == 'MIMO':
        b_cbf = torch.kron(at_0, ar_0)
    else:
        b_cbf = ar_0 / torch.abs(ar_0)

    # MVDR
    w_mvdr = model._compute_mvdr_weights(at_0, at_j_list)
    w_mvdr = w_mvdr / (torch.abs(w_mvdr) + 1e-2)
    
    if model.mode == 'MIMO':
        v_0 = torch.kron(at_0, ar_0)
        v_j_list = [torch.kron(aj, arj) for aj, arj in zip(at_j_list, ar_j_list)]
        b_mvdr = model._compute_mvdr_weights(v_0, v_j_list)
    else:
        b_mvdr = model._compute_mvdr_weights(ar_0, ar_j_list)
        
    def get_metrics(w, b):
        if model.env.array_type == 'Linear':
            th = torch.linspace(-np.pi/2, np.pi/2, 360, device=device, dtype=torch.float32)
            ph = torch.zeros_like(th)
        else:
            th_grid = torch.linspace(-np.pi, np.pi, 360, device=device, dtype=torch.float32)
            ph_grid = torch.linspace(-np.pi/2, np.pi/2, 180, device=device, dtype=torch.float32)
            T, P = torch.meshgrid(th_grid, ph_grid, indexing='ij')
            th = T.flatten()
            ph = P.flatten()

        at_scan = model.env.steering_vector(pos_t, th, ph)
        ar_scan = model.env.steering_vector(pos_r, th, ph)
        
        if model.mode == 'PA':
            ec = torch.abs(torch.matmul(w.conj(), at_scan))**2
        else:
            ec = torch.ones_like(th)
        
        if model.mode == 'MIMO':
             w_ex = w.unsqueeze(1)
             at_w = at_scan * w_ex
             v_v = torch.einsum('mk,nk->mnk', at_w, ar_scan).reshape(-1, th.shape[0])
             er = torch.abs(torch.matmul(b.conj(), v_v))**2
        else:
             er = torch.abs(torch.matmul(b.conj(), ar_scan))**2
        
        pat = ec * er
        
        idx_t = torch.argmin(torch.abs(th - target[0]) + torch.abs(ph - target[1]))
        p_t = pat[idx_t]
        p_j = 0.0
        for j in jammers:
            idx_j = torch.argmin(torch.abs(th - j[0]) + torch.abs(ph - j[1]))
            p_j += pat[idx_j]
        sinr = 10*torch.log10(p_t / (p_j + 0.1))
        return pat.detach().cpu().numpy(), sinr.item()

    pat_cbf, sinr_cbf = get_metrics(w_cbf, b_cbf)
    pat_mvdr, sinr_mvdr = get_metrics(w_mvdr, b_mvdr)
    
    return {'cbf': (pat_cbf, sinr_cbf), 'mvdr': (pat_mvdr, sinr_mvdr)}

# ==========================================
# 5. 可视化函数 (增强版)
# ==========================================
def visualize_results(model, loss_history, sinr_history, target, jammers, pattern_data, title):
    joint_pattern, theta, phi = pattern_data
    pattern_db = 10 * np.log10(joint_pattern.detach().cpu().numpy() + 1e-12)
    pattern_db -= np.max(pattern_db)
    
    fig = plt.figure(figsize=(18, 5))
    plt.suptitle(f"Graph-Embedded Optimization: {title}", fontsize=14)

    # Loss & SINR
    ax1 = fig.add_subplot(1, 3, 1)
    ax1_twin = ax1.twinx()
    l1, = ax1.plot(loss_history, 'b-', label='Loss')
    l2, = ax1_twin.plot(sinr_history, 'r--', label='SINR (dB)')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss", color='b')
    ax1_twin.set_ylabel("SINR (dB)", color='r')
    ax1.legend([l1, l2], ['Loss', 'SINR'], loc='center right')
    ax1.grid(True, alpha=0.3)

    # Pattern
    ax3 = fig.add_subplot(1, 3, 3)
    if model.env.array_type == 'Linear':
        deg = np.rad2deg(theta.cpu().numpy())
        ax3.plot(deg, pattern_db, 'k-', linewidth=2, label='Proposed')
        t_deg = np.rad2deg(target[0])
        ax3.axvline(t_deg, color='g', linestyle='--', label='Target')
        for idx, jam in enumerate(jammers):
            j_deg = np.rad2deg(jam[0])
            ax3.axvline(j_deg, color='r', linestyle=':', label=f'Jammer {idx+1}')
        ax3.set_ylim(-60, 0)
        ax3.set_xlabel("Angle (Deg)")
        ax3.legend()
    else:
        shape_t, shape_p = 360, 180
        if theta.numel() != shape_t * shape_p:
            print(f"⚠️ Warning: Data size {theta.numel()} mismatch. Plotting skipped.")
            return

        T_np = theta.cpu().numpy().reshape(shape_t, shape_p)
        P_np = phi.cpu().numpy().reshape(shape_t, shape_p)
        Z = pattern_db.reshape(shape_t, shape_p)
        
        U = np.sin(T_np) * np.cos(P_np)
        V = np.sin(T_np) * np.sin(P_np)
        
        c = ax3.pcolormesh(U, V, Z, cmap='jet', vmin=-60, vmax=0, shading='auto')
        fig.colorbar(c, ax=ax3, label='Gain (dB)')
        
        t_u = np.sin(target[0]) * np.cos(target[1])
        t_v = np.sin(target[0]) * np.sin(target[1])
        ax3.plot(t_u, t_v, 'w*', markersize=15, markeredgecolor='k', label='Target')
        
        for idx, jam in enumerate(jammers):
            j_u = np.sin(jam[0]) * np.cos(jam[1])
            j_v = np.sin(jam[0]) * np.sin(jam[1])
            ax3.plot(j_u, j_v, 'rx', markersize=10, markeredgecolor='k', label=f'Jammer {idx+1}')
            
        ax3.set_xlabel("U = sin(θ)cos(φ)")
        ax3.set_ylabel("V = sin(θ)sin(φ)")
        ax3.legend()
        
    ax3.set_title("Resulting Pattern")
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. 主仿真函数 (Modified)
# ==========================================
def run_simulation(case_name, mode, array_type, target_deg, jammers_deg, init_file):
    print(f"\n>>> 启动仿真: {case_name}")
    
    if array_type.lower() == 'linear':
        target = (np.deg2rad(target_deg[0]), 0)
        jammers = [(np.deg2rad(j[0]), 0) for j in jammers_deg]
    else:
        target = (np.deg2rad(target_deg[0]), np.deg2rad(target_deg[1]))
        jammers = [(np.deg2rad(j[0]), np.deg2rad(j[1])) for j in jammers_deg]

    env = RadarEnvironment(M=7, N=7, array_type=array_type)
    model = KPDNet_Optimizer(env, mode=mode, target_angle=target, jammer_angles=jammers, init_file=init_file).to(device)

    params_weights = [
        model.w_real, model.w_imag, model.b_real, model.b_imag,
        *list(model.net.parameters())
    ]
    opt_weights = torch.optim.Adam(params_weights, lr=0.005)
    opt_geometry = torch.optim.Adam([model.L_t, model.L_r], lr=0.001) 

    scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(opt_weights, T_max=400, eta_min=1e-5)

    best_sinr = -1e9
    best_state = copy.deepcopy(model.state_dict())
    rollback_count = 0
    loss_history = []
    sinr_history = []
    
    for k in range(500):
        opt_weights.zero_grad()
        opt_geometry.zero_grad()
        
        loss, pattern, theta, phi, sinr = model.forward_pass(target, jammers)
        
        if k > 20 and (best_sinr - sinr.item() > 5.0):
            print(f"  ⚠️ Iter {k}: SINR崩盘 ({sinr.item():.2f} < Best {best_sinr:.2f}). 执行回滚!")
            model.load_state_dict(best_state)
            for param_group in opt_weights.param_groups:
                param_group['lr'] *= 0.5
            rollback_count += 1
            continue

        if sinr.item() > best_sinr:
            best_sinr = sinr.item()
            best_state = copy.deepcopy(model.state_dict())

        loss.backward()
        
        opt_weights.step()
        scheduler_w.step()
        
        if k % 10 == 0 and k < 300: 
            opt_geometry.step()
            model.update_geometry_constrained()

        loss_history.append(loss.item())
        sinr_history.append(sinr.item())

        if k % 50 == 0:
            print(f"  Iter {k}: SINR={sinr.item():.2f} dB (Best: {best_sinr:.2f})")

    print(f"\n✅ 仿真结束. 加载最优模型 (SINR: {best_sinr:.2f} dB)")
    model.load_state_dict(best_state)
    
    # --- 计算结果并保存 ---
    _, pat_final, theta_scan, phi_scan, sinr_final = model.forward_pass(target, jammers)
    benchmarks = calculate_benchmarks(model, target, jammers)
    
    mat_filename = f"Result_{case_name.replace(' ', '_')}.mat"
    
    # ================= 修改开始 =================
    # 在保存字典中增加 loss_history 和 sinr_history
    save_data = {
        'theta': np.rad2deg(theta_scan.detach().cpu().numpy()),
        'phi': np.rad2deg(phi_scan.detach().cpu().numpy()),
        'pattern_proposed': 10*np.log10(pat_final.detach().cpu().numpy() + 1e-12),
        'pattern_mvdr': 10*np.log10(benchmarks['mvdr'][0] + 1e-12),
        'pattern_cbf': 10*np.log10(benchmarks['cbf'][0] + 1e-12),
        'sinr_proposed': sinr_final.item(),
        'sinr_mvdr': benchmarks['mvdr'][1],
        'sinr_cbf': benchmarks['cbf'][1],
        'L_t_opt': model.L_t.detach().cpu().numpy(),
        'L_r_opt': model.L_r.detach().cpu().numpy(),
        'loss_history': np.array(loss_history),
        'sinr_history': np.array(sinr_history)
    }
    
    savemat(mat_filename, save_data)
    print(f"💾 数据已保存至 {mat_filename}")
    
    visualize_results(model, loss_history, sinr_history, target, jammers, (pat_final, theta_scan, phi_scan), case_name)

if __name__ == "__main__":
    # 场景 1: 多干扰抑制 (Linear MIMO)
    run_simulation(
        "Linear_MIMO_MultiJammer", 'MIMO', 'Linear', 
        target_deg=(10, 0), 
        jammers_deg=[(-40, 0), (-20, 0), (50, 0)],
        init_file='init_linear.mat'
    )
    
    # # 场景 2: 主瓣干扰 (Linear PA)
    run_simulation(
        "Linear_MIMO_MainlobeJammer", 'MIMO', 'Linear',
        target_deg=(10, 0),
        jammers_deg=[(13, 0)],
        init_file='init_linear.mat'
    )
    