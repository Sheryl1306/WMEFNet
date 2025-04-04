# WMEFNet
## Abstract
Existing methods often fail to sufficiently focus on edge information, which is essential for precise object localization in aerial imagery. Aerial imagery presents unique challenges due to factors such as the dense distribution of small objects and significant variations in object scales, which are difficult for conventional detection frameworks to address. To address these challenges, we propose a novel and effective object detection method specifically designed for aerial imagery, namely the Wavelet-Enhanced Multi-Scale Edge Fusion Network (WMEFNet). WMEFNet integrates an edge information extractor and cross-channel fusion module within its backbone network, facilitating comprehensive edge feature extraction. Additionally, it introduces the Wavelet-Context Fusion Pyramid Network (WCFPN) to aggregate multi-level edge information flows and enable feature fusion across diverse receptive fields. We further present the Wavelet Upsampling Feature Fusion Module and Wavelet Downsampling Module, which employ wavelet transforms to decouple high-frequency and low-frequency features, guiding multi-level feature fusion and preserving fine details. Extensive experimental evaluations on multiple benchmark datasets demonstrate that WMEFNet achieves superior detection accuracy, offering an efficient and precise solution for object detection challenges in aerial imagery.


![image](https://github.com/user-attachments/assets/f46004e1-19a7-4c6b-bbe9-2368fabc6359)


## Detailed Instructions
RT-DETR Environment Setup
1.Execute pip uninstall ultralytics to completely uninstall the ultralytics library from your environment. 

2.After uninstalling, execute the command again. If you see the message "WARNING: Skipping ultralytics as it is not installed," it indicates that the library has been successfully removed.

3.If you need to use the official CLI method, you'll need to install the ultralytics library. 

4.Additional required package installation command:
pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 

## User Guide
To train with your own dataset, simply add your dataset path following the annotation format, and you can begin training.


The code is in the master file.

## Declaration
This code project will be published as a paper in *The Visual Computer*. The upload to GitHub is done solely to meet the journal's requirements for increased transparency and reproducibility, with the algorithm's source code being made publicly available whenever feasible. Currently, the paper is in the submission stage, and the publicly available source code is fully functional, but it is for reference only. Any unauthorized use or modification of the code will be subject to legal action. After the paper is accepted, the full code will be openly available for readers' learning and use.
