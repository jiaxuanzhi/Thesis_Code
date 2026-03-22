%% AI for waveform and filter design 
% @Yizhen Jia, UESTC
% 2024/08/29
%%
close all
clear

Ka = 1;%sequence number
num_train = 400;  
LearnRate = 0.0005;  
Los_Total = zeros(Ka,num_train); 
MetricTal=cell(1,Ka);
for idx = Ka

M = 3;
L = 2^(8);
Num_chop = 4; 
Len_SingleChop = ceil(L/Num_chop/4);% ceil(L/Num_chop/4)

Parameter.L = L;
Parameter.M = M;
Parameter.Num_chop = Num_chop;
Parameter.Len_SingleChop = Len_SingleChop;

Construct_Proposed_Net_First
% Construct_Proposed_Net_without_PRETrain

net1 = net; %第1条波形滤波器组
HRnet1 = net;
HInet1 = net;

net2 = net;  %第2条波形滤波器组
HRnet2 = net;
HInet2 = net;

net3 = net;  %第3条波形滤波器组
HRnet3 = net;
HInet3 = net;
%初始化随机输入序列
inputs1 = rand(1024,1,1); %cos(2*pi*rand(1024,1,1))
inputs1 = dlarray(inputs1,"SSC");
inputs2 = 2*rand(1024,1,1)-1; %cos(2*pi*rand(1024,1,1)),2*rand(1024,1,1)-1
inputs2 = dlarray(inputs2,"SSC");
inputs3 = randn(1024,1,1); %cos(2*pi*rand(1024,1,1)),2*rand(1024,1,1)-1
inputs3 = dlarray(inputs3,"SSC");


% 定义损失函数
gradientsAvg11 = [];  %第1条波形中间梯度变量
squaredGradientsAvg11 = [];
gradientsAvg12 = [];
squaredGradientsAvg12 = [];
gradientsAvg13 = [];
squaredGradientsAvg13 = [];

gradientsAvg21 = [];   %第2条波形中间梯度变量
squaredGradientsAvg21 = [];
gradientsAvg22 = [];
squaredGradientsAvg22 = [];
gradientsAvg23 = [];
squaredGradientsAvg23 = [];

gradientsAvg31 = [];   %第3条波形中间梯度变量
squaredGradientsAvg31 = [];
gradientsAvg32 = [];
squaredGradientsAvg32 = [];
gradientsAvg33 = [];
squaredGradientsAvg33 = [];

iter = 0;
for i=1:num_train
    % 进行训练 
        % 更新网络
[loss,Gradnet1,GradHRnet1,GradHInet1,Gradnet2,GradHRnet2,GradHInet2,Gradnet3,GradHRnet3,GradHInet3,...
    state_1,HRstate_1,HIstate_1,state_2,HRstate_2,HIstate_2,state_3,HRstate_3,HIstate_3,...
    Amp_Cor,Amp_Jammer,Waveform_Result,Metric_waveform] ...
    = dlfeval(@modelLoss_CI,net1,HRnet1,HInet1,net2,HRnet2,HInet2,net3,HRnet3,HInet3,...
    inputs1,inputs2,inputs3,Parameter);  % 
        % 可以跟踪输入在网络中的计算过程，从而得到梯度信息Gradients
        net1.State = state_1;
        HRnet1.State = HRstate_1;
        HInet1.State = HIstate_1;

        net2.State = state_2;
        HRnet2.State = HRstate_2;
        HInet2.State = HIstate_2;

        net3.State = state_3;
        HRnet3.State = HRstate_3;
        HInet3.State = HIstate_3;
        iter = iter + 1;
        [net1,gradientsAvg11,squaredGradientsAvg11] = adamupdate(...
                     net1,Gradnet1,gradientsAvg11,squaredGradientsAvg11,iter,LearnRate);
        [HRnet1,gradientsAvg12,squaredGradientsAvg12] = adamupdate(...
            HRnet1,GradHRnet1,gradientsAvg12,squaredGradientsAvg12,iter,LearnRate);
        [HInet1,gradientsAvg13,squaredGradientsAvg13] = adamupdate(...
            HInet1,GradHInet1,gradientsAvg13,squaredGradientsAvg13,iter,LearnRate);

        [net2,gradientsAvg21,squaredGradientsAvg21] = adamupdate(...
            net2,Gradnet2,gradientsAvg21,squaredGradientsAvg21,iter,LearnRate);
        [HRnet2,gradientsAvg22,squaredGradientsAvg22] = adamupdate(...
            HRnet2,GradHRnet2,gradientsAvg22,squaredGradientsAvg22,iter,LearnRate);
        [HInet2,gradientsAvg23,squaredGradientsAvg23] = adamupdate(...
            HInet2,GradHInet2,gradientsAvg23,squaredGradientsAvg23,iter,LearnRate);

        [net3,gradientsAvg31,squaredGradientsAvg31] = adamupdate(...
            net3,Gradnet3,gradientsAvg31,squaredGradientsAvg31,iter,LearnRate);
        [HRnet3,gradientsAvg32,squaredGradientsAvg32] = adamupdate(...
            HRnet3,GradHRnet3,gradientsAvg32,squaredGradientsAvg32,iter,LearnRate);
        [HInet3,gradientsAvg33,squaredGradientsAvg33] = adamupdate(...
            HInet3,GradHInet3,gradientsAvg33,squaredGradientsAvg33,iter,LearnRate);
    % 进行测试
    if mod(i,100)==99
        Amp_Cor_T = extractdata(Amp_Cor);
        Amp_Jammer_T = extractdata(Amp_Jammer);
        figure(1)
        plot(squeeze(Amp_Cor_T(:,1,1)),'r-')       
        hold on
        plot(squeeze(Amp_Cor_T(:,2,2)),'b--')
        plot(squeeze(Amp_Cor_T(:,3,3)),'g-.')
        hold off
        title("AtuoCorrelation of waveform")
        xlabel('time'),ylabel('Amplitude'),legend("Waveform1","Waveform2","Waveform3")

        figure(2)
        plot(squeeze(Amp_Cor_T(:,1,2)),'r-')
        
        hold on
        plot(squeeze(Amp_Cor_T(:,1,3)),'b--')
        plot(squeeze(Amp_Cor_T(:,2,3)),'g-.')
        hold off
        title("CrossCorrelation of waveform")
        xlabel('time'),ylabel('Amplitude'),legend("Waveform1-2","Waveform1-3","Waveform2-3")

        figure(3)
        plot(squeeze(Amp_Jammer_T(:,1,1)),'r-')
        hold on
        plot(squeeze(Amp_Jammer_T(:,1,2)))
        plot(squeeze(Amp_Jammer_T(:,1,3)))
        plot(squeeze(Amp_Jammer_T(:,2,2)))
        plot(squeeze(Amp_Jammer_T(:,2,3)))
        plot(squeeze(Amp_Jammer_T(:,3,3)))
        hold off
        title("CrossCorrelation of Jammerwaveform")
        xlabel('time'),ylabel('Amplitude')

    end

    Los_Total(idx,i)=loss;
    fprintf("第%d轮次训练损失为%f\n",i,loss);
end

MetricTal{idx} = Metric_waveform;

end  

