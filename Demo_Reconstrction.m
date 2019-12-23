
%% IFR_EDMSP
close all; clc;
clear all;
%% Add  path
addpath(genpath('DAEs'))
getd=@(p)path(path,p);
addpath('./data')
addpath('./func')
addpath('./models')
addpath('./quality_assess')
addpath('./mask')
addpath('./sub')
%% select denoiser
denoiser_name = 'caffe'; % make sure matCaffe is installed and its location is added to path
%% set to 0 if you want to run on CPU (very slow)
gpu = 1;
if gpu 
    gpuDevice(gpu); 
end
%% add mask
load('mask_random_015.mat')
Q1=mask;
figure(356); imshow(fftshift(Q1),[]);
n = size(Q1,2);m = size(Q1,1);
k = sum(sum(Q1));
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, k,1-k/n/n);
fprintf(1, 'R=%f\n', n*n/k);
%% read data
im = double(imread('boats.tif'));
M0 = double(im);
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
if (max(M0(:))<2);   M0 = M0*255;    end
%% Network parameter setting
params.sigma_net = 3;   
params2.sigma_net = 5; 
params3.sigma_net = 8;  
%net1
load('MWCNN_GDSigma3_3D_400-epoch-45.mat');  net1 = net;
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
load('MWCNN_GDSigma5_3D_400-epoch-45.mat');  net2 = net;
net2 = dagnn.DagNN.loadobj(net2) ;
net2.removeLayer('objective') ;
out_idx = net2.getVarIndex('prediction') ;
net2.vars(net2.getVarIndex('prediction')).precious = 1 ;
net2.mode = 'test';
if gpu
    net2.move('gpu'); 
end
params2.out_idx = out_idx;  params2.gpu = gpu;
% net2
load('MWCNN_GDSigma8_3D_400-epoch-40.mat');  net3 = net;
net3 = dagnn.DagNN.loadobj(net3) ;
net3.removeLayer('objective') ;
out_idx = net3.getVarIndex('prediction') ;
net3.vars(net3.getVarIndex('prediction')).precious = 1 ;
net3.mode = 'test';
if gpu
    net3.move('gpu'); 
end
params3.out_idx = out_idx;  params3.gpu = gpu;
clear net;

%% reconstrction
[res,I1,PSNRS,SSIMS]  = Res_test_multi_3sigma(M0 , Q1,params,  params2,params3, net1, net2,net3 );
%% show result
[psnr4, ssim4, ~] = MSIQA(res*255, abs(I1)*255);
figure(333);subplot(131);imshow(im,[]); title('Ground Truth')
subplot(132);imshow(M0,[]); title('undersampled')
subplot(133);imshow(res,[]); title('reconstrcted')