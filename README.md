# IFR-EDMSP 
**Paper**: Iterative Feature Refinement with Network-driven Prior for Image Restoration  
**Authors**: Miaomiao Meng, Yiling Liu, Mengting Li, Qiegen Liu, Minghui Zhang

## Motivation
Image restoration (IR) has been extensively studied with lots of promising strategies accumulated over the years. However, most existing methods still have large rooms to improve. In this work, we boost an unsupervised iterative feature refinement model (IFR) with the enhanced high-dimensional deep mean-shift prior (EDMSP) for IR tasks, dubbed IFR-EDMSP. The proposed model inherits the fantastic noise suppression character of embedded network and the fine details preservation ability of IFR model. In addition, based on the facts that multiple implementation of artificial noise in prior learning favors better underlying representation capability, three-sigma rule is adopted in this model to obtain more robust result. Extensive experiments on compressed sensing, image deblurring and super-resolution verified the effectiveness of the proposed method.

The overall REDAEP algorithm is as follows:
<div align="center">
  
<img src="https://github.com/yqx7150/IFR-EDMSP/blob/master/algorithm.png" width = "400" height = "450">  
  
 </div>

The flowchart illustration of IFR-EDMSP for image restoration
![repeat-IFR-EDMSP](https://github.com/yqx7150/IFR-EDMSP/blob/master/iter.png)



