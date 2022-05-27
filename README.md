# TRT-Hackathon-2022-final
&lt;Good Luck To You!> 's work for &lt;TRT-Hackathon-2022-final>

---
## 总述
本次复赛我们选择使用TensorRT优化部署的模型是[MobileVit](https://arxiv.org/abs/2110.02178)，该项工作由Apple的研究者发表在ICLR2022上，项目开源在[地址](https://github.com/apple/ml-cvnets)。
## 原始模型
### 模型简介
MobileVit结合了CNN和ViT的优势构建了一个轻量级、低延时和移动设备友好的通用基础网络模型。轻量卷积神经网络(CNN)是移动视觉任务的实际应用。CNN的空间归纳偏差允许他们在不同的视觉任务中以较少的参数学习表征，然而CNN在空间上是却是局部的；为了学习全局表征，MobileVit引入了基于自注意力的**轻量化**的Vision Transformer(ViTs)模块，轻量化主要体现在使用了更小的注意力模块参数L与d。MobileVit能够作为检测、分割等视觉任务的基础网络应用于移动设备之中，均易于部署且能收获到不错的精度。\
模型的整体结构如下图所示,其中MV2代表MobileNetV2 block：
![网络结构图](./Resources/MobileViT_arch.png) \
MobileVit共有三种不同的网络规模(XXS，XS，S)，参数量逐渐升高，分别为1.3M，2.3M，5.6M，具体如下图所示:
![不同尺寸结果](./Resources/MobileViT_size.png) \
在MS-COCO数据集与ImageNet数据集上对比各个轻量化网络模型的测试结果如下图所示:
![实验结果](./Resources/result.png)
### 模型优化的难点
当前针对于MobileVit-S网络模型的优化难点总结如下：
1. 对于动态batch size的支持。Apple提供的pt与config文件无法直接生成动态batch size的TensorRT engine，需要手动调整源代码中的网络结构以及对算子图进行优化。
2. attention、layernorm等模块TensorRT plugin的实现，数据格式支持FP32、TF32以及FP16。
3. int8以及int8相关TensorRT plugin的实现。轻量化模型在移动端使用int8数据格式能获得更好的加速效果。