function [loss,Gradnet1,GradHRnet1,GradHInet1,Gradnet2,GradHRnet2,GradHInet2,Gradnet3,GradHRnet3,GradHInet3,...
    state_1,HRstate_1,HIstate_1,state_2,HRstate_2,HIstate_2,state_3,HRstate_3,HIstate_3,...
    Amp_Cor,Amp_Jammer2,Waveform_Result,Metric_waveform]...
    = modelLoss_CI(net1,HRnet1,HInet1,net2,HRnet2,HInet2,net3,HRnet3,HInet3,...
    inputs1,inputs2,inputs3,Parameter)
%   update parameters
L = Parameter.L;
M = Parameter.M;
Num_chop = Parameter.Num_chop;
Len_SingleChop = Parameter.Len_SingleChop;

[out_1,state_1] = forward(net1, inputs1);  % 网络前向传播获得输出 
[HRout_1,HRstate_1] = forward(HRnet1, inputs1);  % 网络前向传播获得H的实部输出 
[HIout_1,HIstate_1] = forward(HInet1, inputs1);  % 网络前向传播获得H的实部输出 

[out_2,state_2] = forward(net2, inputs2);  % 网络前向传播获得输出 
[HRout_2,HRstate_2] = forward(HRnet2, inputs2);  % 网络前向传播获得H的实部输出 
[HIout_2,HIstate_2] = forward(HInet2, inputs2);  % 网络前向传播获得H的实部输出 

[out_3,state_3] = forward(net3, inputs3);  % 网络前向传播获得输出 
[HRout_3,HRstate_3] = forward(HRnet3, inputs3);  % 网络前向传播获得H的实部输出 
[HIout_3,HIstate_3] = forward(HInet3, inputs3);  % 网络前向传播获得H的实部输出

out_1 = dlarray(out_1,"SSC");
out_2 = dlarray(out_2,"SSC");
out_3 = dlarray(out_3,"SSC");

out_1 = out_1./max(out_1(:));
out_2 = out_2./max(out_2(:));%归一化输出
out_3 = out_3./max(out_3(:));%归一化输出

HIout_1 = dlarray(HIout_1,"SSC");
HIout_2 = dlarray(HIout_2,"SSC");
HIout_3 = dlarray(HIout_3,"SSC");

HIout_1 = HIout_1./max(HIout_1(:));
HIout_2 = HIout_2./max(HIout_2(:));%归一化输出
HIout_3 = HIout_3./max(HIout_3(:));%归一化输出



HRout_1 = HRout_1./max(HRout_1(:));
HRout_2 = HRout_2./max(HRout_2(:));%归一化输出
HRout_3 = HRout_3./max(HRout_3(:));%归一化输出

NoisePower = sum(HRout_3.^2+HRout_2.^2+HRout_1.^2);

HRout_1 = dlarray(HRout_1,"SSC");
HRout_2 = dlarray(HRout_2,"SSC");
HRout_3 = dlarray(HRout_3,"SSC");

HFilter1_R = HRout_1.*cos(2*pi*HIout_1);
HFilter1_I = HRout_1.*sin(2*pi*HIout_1);
HFilter2_R = HRout_2.*cos(2*pi*HIout_2);
HFilter2_I = HRout_2.*sin(2*pi*HIout_2);
HFilter3_R = HRout_3.*cos(2*pi*HIout_3);
HFilter3_I = HRout_3.*sin(2*pi*HIout_3);


HFilter_R = cat(4,HFilter1_R,HFilter2_R,HFilter3_R);
HFilter_I = cat(4,HFilter1_I,HFilter2_I,HFilter3_I);

Yout = [out_1,out_2,out_3];
out_R = cos(2*pi*Yout);
out_I = sin(2*pi*Yout);

%  计算匹配滤波结果
bias = zeros(M,1);%numFiltersPerGroup*numGroups
RR = dlconv(out_R,HFilter_R,bias,Padding=[L-1,0]);%Real*Real
II = dlconv(out_I,HFilter_I,bias,Padding=[L-1,0]);%Imag*Imag
RI = dlconv(out_R,HFilter_I,bias,Padding=[L-1,0]);%Real*Imag
IR = dlconv(out_I,HFilter_R,bias,Padding=[L-1,0]);%Imag*Real

R_total = RR+II;
I_total = IR-RI;
Amp_Cor = R_total.^2+I_total.^2;

Auto_Matrix = Amp_Cor(:,1,1)+Amp_Cor(:,2,2)+Amp_Cor(:,3,3);
Temp_A = Auto_Matrix(L-1:L+1);
Ener = sum(Auto_Matrix(:));
AISL = Ener-sum(Temp_A(:));% 发射波形自相关的积分旁瓣比,指标2
C_Matrix = Auto_Matrix;
C_Matrix(L-1:L+1)=0;
APSL = max(C_Matrix(:)); %%指标4


CISL = sum(Amp_Cor(:))-Ener;% 发射波形互相关能量，指标3
C2_Matrix = Amp_Cor;
C2_Matrix(L-1:L+1,1,1)=0;
C2_Matrix(L-1:L+1,2,2)=0;
C2_Matrix(L-1:L+1,3,3)=0;
CPL = max(C2_Matrix(:)); %%指标5

Ttem_1 = Amp_Cor(L,1,1);
M_loss1 = abs(sum(Ttem_1(:))-L.^2);%各波形的主瓣能量损失，指标1
Ttem_2 = Amp_Cor(L,2,2);
M_loss2 = abs(sum(Ttem_2(:))-L.^2);%各波形的主瓣能量损失，指标1
Ttem_3 = Amp_Cor(L,3,3);
M_loss3 = abs(sum(Ttem_3(:))-L.^2);%各波形的主瓣能量损失，指标1


Index = (1:Len_SingleChop)' + ceil(L/Num_chop)*(0:Num_chop-1);
Index = Index(:);
JameroutR = dlarray(zeros(L,M,1),"SSC");
JameroutI = dlarray(zeros(L,M,1),"SSC");
JameroutR(Index,:,:) = out_R(Index,:,:);
JameroutI(Index,:,:) = out_I(Index,:,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% C\&F jamming %%%%%%%%%%%%%%

% Jamer1 = circshift(Jamer,Len_SingleChop,1);
% Jamerout_R = cos(2*pi*Jamer1);
% Jamerout_I = sin(2*pi*Jamer1);
% 
% JamerRR = dlconv(Jamerout_R,HFilter_R,bias,Padding=[L-1,0]);%Real*Real
% JamerII = dlconv(Jamerout_I,HFilter_I,bias,Padding=[L-1,0]);%Imag*Imag
% JamerRI = dlconv(Jamerout_R,HFilter_I,bias,Padding=[L-1,0]);%Real*Imag
% JamerIR = dlconv(Jamerout_I,HFilter_R,bias,Padding=[L-1,0]);%Imag*Real
% Amp_Jammer = (JamerRR+JamerII).^2+(JamerIR-JamerRI).^2;
% Jammer_Loss = sum(Amp_Jammer(:));% 转发式干扰波形滤波后的能量，指标4
% IPL = max(Amp_Jammer(:)); %%指标6

%%%%%%%%%%%%%%%%%%%%%%%%%% C\&I jamming %%%%%%%%%%%%%%

Jamerout_R = JameroutR + circshift(JameroutR,Len_SingleChop,1)+circshift(JameroutR,2*Len_SingleChop,1)...
    + circshift(JameroutR,3*Len_SingleChop,1);
Jamerout_I = JameroutI + circshift(JameroutI,Len_SingleChop,1)+circshift(JameroutI,2*Len_SingleChop,1)...
    + circshift(JameroutI,3*Len_SingleChop,1);

JamerRR = dlconv(Jamerout_R,HFilter_R,bias,Padding=[L-1,0]);%Real*Real
JamerII = dlconv(Jamerout_I,HFilter_I,bias,Padding=[L-1,0]);%Imag*Imag
JamerRI = dlconv(Jamerout_R,HFilter_I,bias,Padding=[L-1,0]);%Real*Imag
JamerIR = dlconv(Jamerout_I,HFilter_R,bias,Padding=[L-1,0]);%Imag*Real
Amp_Jammer2 = (JamerRR+JamerII).^2+(JamerIR-JamerRI).^2;
Jammer_Loss2 = sum(Amp_Jammer2(:));% 转发式干扰波形滤波后的能量，指标4
IPL2 = max(Amp_Jammer2(:)); %%指标6

%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%

loss = (M_loss1+M_loss2+M_loss3)*0.1 + AISL*0.3/(L*M) + CISL*0.3/(M*L) + Jammer_Loss2*0.3/(L*M)...
    +APSL*0.2 + CPL*0.3 + IPL2*0.5 + NoisePower/(L*L); % 
Metric_waveform.M_loss = M_loss1+M_loss2+M_loss3;
Metric_waveform.AISL  = AISL;
Metric_waveform.APSL  = APSL;
Metric_waveform.CISL  = CISL;
Metric_waveform.CPL  = CPL;
Metric_waveform.Jammer_Loss  = Jammer_Loss2;
Metric_waveform.IPL  = IPL2;


Waveform_Result = cat(3,cat(2,HFilter1_R,HFilter2_R,HFilter3_R)+1i*...
    cat(2,HFilter1_I,HFilter2_I,HFilter3_I),exp(1i*2*pi*Yout));% 

[Gradnet1,GradHRnet1,GradHInet1,...
    Gradnet2,GradHRnet2,GradHInet2,...
        Gradnet3,GradHRnet3,GradHInet3,] = dlgradient(loss,net1.Learnables,HRnet1.Learnables,HInet1.Learnables,...
            net2.Learnables,HRnet2.Learnables,HInet2.Learnables,...
                net3.Learnables,HRnet3.Learnables,HInet3.Learnables);  % 
end
