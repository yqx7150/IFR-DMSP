clear; %clc;

%% load path
addpath(genpath('DAEs'))
addpath('./data')
addpath('./func')
addpath('./models')
addpath('./quality_assess')
addpath('./sub')
%% read images
ext = {'*.bmp'};
folder = './data/Set5';
for ii = 1 : length(ext)
    filePaths = dir(fullfile(folder, ext{ii}));
end
%%
showresult  = 1;
scale      = 3 ;  %2,3,4
%% choose gpu
gpu = 1;
if gpu > 0
    gpuDevice(gpu);
end
%% add models
params.sigma_net  = 5;
params.num_iter   = 1000;  
params2.sigma_net = 8;
params2.num_iter  = 1000;
params3.sigma_net  = 10;
params3.num_iter   = 1000;  
% net1
load('MWCNN_GDSigma5_3D_400-epoch-45');
net1 = net;
net1 = dagnn.DagNN.loadobj(net1) ;
net1.removeLayer('objective') ;
out_idx = net1.getVarIndex('prediction') ;
net1.vars(net1.getVarIndex('prediction')).precious = 1 ;
net1.mode = 'test';
if gpu
    net1.move('gpu');
end
params.out_idx = out_idx;  params.gpu = gpu;
clear net;
% net2
load('MWCNN_GDSigma8_3D_400-epoch-40');net2 = net;
net2 = dagnn.DagNN.loadobj(net2) ;
net2.removeLayer('objective') ;
out_idx = net2.getVarIndex('prediction') ;
net2.vars(net2.getVarIndex('prediction')).precious = 1 ;
net2.mode = 'test';
if gpu
    net2.move('gpu');
end
params2.out_idx = out_idx;  params2.gpu = gpu;
clear net;
% net3
load('MWCNN_GDSigma10_3D_400-epoch-45.mat');
net3 = net;
net3 = dagnn.DagNN.loadobj(net3) ;
net3.removeLayer('objective') ;
out_idx = net3.getVarIndex('prediction') ;
net3.vars(net3.getVarIndex('prediction')).precious = 1 ;
net3.mode = 'test';
if gpu
    net3.move('gpu');
end
params3.out_idx = out_idx;  params3.gpu = gpu;


%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
times = zeros(1,length(filePaths));
%% SISR
for i = 1 : length(filePaths)
    im = imread(fullfile(folder, filePaths(i).name));
    if scale == 3
        im  = modcrop(im, 8*scale);
    else
        im  = modcrop(im, 8);
        
    end
    label = double(im);
    params.gt = label;
    degraded  = imresize(label , 1/scale, 'bicubic');
    %%
    [output, pnsr1, ssim1 ]   = SISR_3_Net(degraded, scale, params, params2, params3, net1, net2, net3);
    
    
    [PSNRCur, SSIMCur] = MSIQA(label, output);
    
    if showresult
        figure(11),imshow(uint8(output),[]);
        figure(33),plot(pnsr1);
        drawnow;
    end
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end
fprintf('PSNR / SSIM : %.02f / %0.4f, %0.4f.\n', mean(PSNRs),mean(SSIMs));