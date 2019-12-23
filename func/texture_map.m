function [mssim, ssim_map] = ssim(img1, C2)

% ========================================================================
% SSIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2009 Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Structural SIMilarity (SSIM) index between two images
%
% Please refer to the following paper and the website with suggested usage
%
% Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
% quality assessment: From error visibility to structural similarity,"
% IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,
% Apr. 2004.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
%
% Note: This program is different from ssim_index.m, where no automatic
% downsampling is performed. (downsampling was done in the above paper
% and was described as suggested usage in the above website.)
%
% Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size
%            depends on the window size and the downsampling factor.
%
%Basic Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim, ssim_map] = ssim(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim, ssim_map] = ssim(img1, img2, K, window, L);
%
%Visualize the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%========================================================================



% [M ,N] = size(img1);

img1 = double(img1);
%img2 = double(img2);

% blur_mask=fspecial('gaussian',25,5.5);
% blur_mask=ones(5,5);
%blur_mask=blur_mask/sum(blur_mask(:));
% img2=filter2(blur_mask, img1, 'same');

sigma=50; %50; %50 %55.5;
% sigma2=50;
ksize=bitor(round(3*sigma),1);
% ksize=min(M/2,ksize);
% ksize=min(N/2,ksize);
blur_mask=fspecial('gaussian',ksize,sigma);
%blur_mask=blur_mask/sum(blur_mask(:));
img2=filter2(blur_mask, img1, 'same');

% g=fspecial('gaussian',[5,5],sigma);
% % s=fspecial('gaussian',[5,5],sigma2);
% img2=conv2(img1, g, 'same');
% im2=conv2(img1, s, 'same');


% automatic downsampling
%f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter
%if(f>1)
%    lpf = ones(f,f);
%    lpf = lpf/sum(lpf(:));
%    img1 = imfilter(img1,lpf,'symmetric','same');
%    img2 = imfilter(img2,lpf,'symmetric','same');
%
%    img1 = img1(1:f:end,1:f:end);
%    img2 = img2(1:f:end,1:f:end);
%end

%C1 = (K(1)*L)^2;
%C2 = C2^2; %(K(2)*L)^2;
% C=2;
window=ones(5,5);
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'same');
mu2   = filter2(window, img2, 'same');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'same') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'same') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'same') - mu1_mu2;

%ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
ssim_map1 = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);

%ssim_map=ssim_map./max(ssim_map(:));
ssim_map=1-abs(ssim_map1);
% ssim_map1=smf(ssim_map1,[min(ssim_map1(:)),0.7*max(ssim_map1(:))]);
%    fx = diff(im2,1,2);
%    fx = padarray(fx, [0 1 0], 'post');
%    fy = diff(im2,1,1);
%    fy = padarray(fy, [1 0 0], 'post');
% Dx=filter2(s,abs(fx),'same');
% Dy=filter2(s,abs(fy),'same');
% Lx=abs(filter2(s,fx,'same'));
% Ly=abs(filter2(s,fy,'same'));
% rtv=(Dx)./(Lx+C)+(Dy)./(Ly+C);
% % rtv=Lx+Ly;
% T2=mat2gray(rtv);
% T2=smf(T2,[min(T2(:)),0.1*max(T2(:))]);
% ssim_map=(ssim_map1+T2)/2;
% ssim_map=smf(ssim_map,[min(ssim_map(:)),0.5*max(ssim_map(:))]);
mssim = mean2(ssim_map);
return
