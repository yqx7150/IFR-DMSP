%% Add  path
addpath(genpath('DAEs'))
getd=@(p)path(path,p);
addpath('./data')
addpath('./func')
addpath('./models')
addpath('./quality_assess')
addpath('./sub')
%% select denoiser
denoiser_name = 'caffe'; % make sure matCaffe is installed and its location is added to path
% denoiser_name = 'matconvnet'; % make sure matconvnet is installed and its location is added to path

%% set to 0 if you want to run on CPU (very slow)
gpu = 1;
if gpu 
    gpuDevice(gpu); 
end

%% load kernels and image 
load('Levin09_modify.mat'); 
gt         = double(imread('boats.tif'));  
kernel     = kernels{2}; %19%17%15%27%13%21%23%%25
sigma_d    = 255 * .01;  %03;01
%%  for some reason Caffe input needs even dimensions...
w           = size(gt,2); w = w - mod(w, 2);
h           = size(gt,1); h = h - mod(h, 2);
gt          = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...

pad         = floor(size(kernel)/2);   
gt_extend   = padarray(gt, pad, 'replicate', 'both');

degraded    = convn(gt_extend, rot90(kernel,2), 'valid');
noise       = randn(size(degraded));
degraded    = degraded + noise * sigma_d;
%% Network parameter setting
params.gt         = gt;        
params.sigma_net  = 3;   
params.num_iter   = 500;
params2.sigma_net = 5;  
params2.num_iter  = 500;
params3.sigma_net = 8;  
params3.num_iter  = 500;
%% net1
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
% % 
%% net2
clear net
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
% net3
clear net
load('MWCNN_GDSigma8_3D_400-epoch-40');  net3 = net;
net3 = dagnn.DagNN.loadobj(net3) ;
net3.removeLayer('objective') ;
out_idx = net3.getVarIndex('prediction') ;
net3.vars(net3.getVarIndex('prediction')).precious = 1 ;
net3.mode = 'test';
if gpu
    net3.move('gpu'); 
end
params3.out_idx = out_idx;  params3.gpu = gpu;

%% non-blind deblurring demo
[restored_extend,PSNRS,SSIMS]  = IFR_DMSP_deblur_multi_3sigma(degraded, kernel, sigma_d, params,  params2,params3, net1, net2,net3 );

%% Output evaluation parameters
restored        = restored_extend(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
[psnr4, ssim4]  = MSIQA(gt, restored);
[psnr4, ssim4]
[maxpsnr,index] = sort(PSNRS);
[maxpsnr,index]
plot(PSNRS);

subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(degraded/255); title('Blurry')
subplot(133);
imshow(restored/255); title('Restored')
