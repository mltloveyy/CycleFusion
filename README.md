# CycleFusion

## Requirements

- python==3.8
- torch==2.1.2+cu11.8
- torchvision==0.16.2+cu11.8
- opencv-python
- scipy
- timm
- einops

## Updates

- **2024/4/21**: 创建项
- **2024/4/25**: 更新data_process.py, 添加测试程序
- **2024/5/23**: 添加质量分数评估接口和测试程序, 完成main.py调试
- **2024/6/17**: 完成main_score.py调试
- **2024/7/16**: 调整data_process.py和quality.py
- **2024/7/26**: 添加分布训练策略
- **2024/7/29**: 优化质量分数权重计算, 添加质量分数评估的SSIM损失
- **2024/8/8**: 改为可学习的融合层
- **2024/8/16**: 添加绘制DET曲线

## Overview

![figure1](figure1.png)

<div align=center>
<sup>Figure 1: Train strategy(left), Framework(right top) and losses(right bottom).</sup>
</div>


## Experimental record
| Date | configs | losses | comments |
| --- | --- | --- | --- |
| 2024-07-19 09:40:14 | batch_size=2, critic=2, data_dir='images/dataset2', epochs=200, fuse_weight=0.7, image_size=256, is_train=True, lr=0.0001, output_dir='output/20240719_094014', patience=5, pretrain_weight='output/models/aaa', quality_thresh=0.5, regular_factor=0.1, save_interval=5, seed=40, ssim_weight=1.0, test_num=20 | quality: mse ssim: ssim pixel: mse fusion: mse | 1. 质量分数损失相较于其他损失下降的最慢; 2. 质量分数预测图过于平滑, 指纹中质量较差部分未表现出来; 3. 在迭代一定epoch后, 融合图像的背景由黑色变为了灰色
| 2024-07-28 13:09:31 | batch_size=2, critic=5, data_dir='images/dataset2', epochs=200, fuse_weight=1.0, image_size=256, is_train=True, lr=0.001, output_dir='output/20240728_130931', patience=10, pixel_weight=1.0, pretrain_weight='output/xxx/models/epochxx', quality_thresh=0.5, quality_weight=1.0, regular_factor=0.1, save_interval=5, seed=40, ssim_weight=1.0, test_num=20, training_encoder=False | quality: mae+ssim restorm: mse+ssim better_fusion: mse+ssim | 1. 质量分数计算方法不准确, 导致TIR和OCT的交接处存在错误判断, 使得某些图像存在明显白色缝隙; 2. 基于1.2*max(q_tir,q_oct)的better_fusion为融合目标的融合图像边缘存在黑色小点(如此设计更加合理); 3. 基于mask(q_tir or q_oct)的better_fusion为融合目标的融合图像接近于Gabor滤波后的指纹; 4. epoch=200时损失仍未收敛
| 2024-08-07 19:09:18 | batch_size=2, critic=5, data_dir='images/dataset5', device=0, epochs=100, fuse_type='pow', fuse_weight=1.0, image_size=256, is_train=True, lr=0.0001, network_type='CDDFuse', output_dir='output/20240807_190917', patience=10, pretrain_weight='output/xxx/models/final', quality_thresh=0.5, quality_weight=1.0, regular_factor=0.1, restore_weight=1.0, save_interval=5, seed=40, ssim_weight=1.0, test_num=20, with_better=True, with_quality=True | quality: mse+10ssim restore: mse+10ssim better_fusion: mse+10ssim | 1. 用可学习的网络代替质量分数来计算融合权重; 2. 质量分数真值舍弃OCL, 改用SC+HD