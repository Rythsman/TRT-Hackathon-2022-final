# TRT-Hackathon-2022-final
&lt;Good Luck To You!> 's work for &lt;TRT-Hackathon-2022-final>

---
## 一、总述
本次复赛我们选择使用TensorRT优化部署的模型是[MobileVit](https://arxiv.org/abs/2110.02178)，该项工作由Apple的研究者发表在ICLR2022上，项目开源在[地址](https://github.com/apple/ml-cvnets)。
## 二、原始模型
### 2.1 模型简介
MobileVit结合了CNN和ViT的优势构建了一个轻量级、低延时和移动设备友好的通用基础网络模型。轻量卷积神经网络(CNN)是移动视觉任务的实际应用。CNN的空间归纳偏差允许他们在不同的视觉任务中以较少的参数学习表征，然而CNN在空间上是却是局部的；为了学习全局表征，MobileVit引入了基于自注意力的**轻量化**的Vision Transformer(ViTs)模块，轻量化主要体现在使用了更小的注意力模块参数L与d。MobileVit能够作为检测、分割等视觉任务的基础网络应用于移动设备之中，均易于部署且能收获到不错的精度。\
模型的整体结构如下图所示,其中MV2代表MobileNetV2 block：
![网络结构图](./Resources/MobileViT_arch.png) \
MobileVit共有三种不同的网络规模(XXS，XS，S)，参数量逐渐升高，分别为1.3M，2.3M，5.6M，具体如下图所示:
![不同尺寸结果](./Resources/MobileViT_size.png) \
在MS-COCO数据集与ImageNet数据集上对比各个轻量化网络模型的测试结果如下图所示:
![实验结果](./Resources/result.png)
### 2.2 模型优化的难点
当前针对于MobileVit-S网络模型的优化难点总结如下：
1. 对于动态batch size的支持。Apple提供的pt与config文件无法直接生成动态batch size的TensorRT engine，需要手动调整源代码中的网络结构以及对算子图进行优化。
2. attention、layernorm等模块TensorRT plugin的实现，数据格式支持FP32、TF32以及FP16；使用Nsys分析实现结果，使用FasterTransformer实现MobileVit网络中的Transformer block插件。
3. int8以及int8相关TensorRT plugin的实现。轻量化模型在移动端使用int8数据格式能获得更好的加速效果。
4. 以MobileVit为基础网络的检测模型相应优化。
## 三、优化过程
下面结合我们的优化流程介绍在过程中遇到的一些问题


### 3.1 ONNX模型导出
修改官网提供的代码导出onnx模型，这个步骤主要是参照官方提供的`main_eval.py`进行实现，通过将`multi_head_attention.py`与`mobilevit_block.py`中reshape操作的`-1`放在第一维度即可支持动态shape的输入。具体原则如下

1. 对于任何用到shape、size返回值的参数时，例如：tensor.view(tensor.size(0), -1)，B,C,H,W = x.shape 这类操作，避免直接使用tensor.size的返回值，而是加上int转换，tensor.view(int(tensor.size(0)), -1), B,C,H,W = map(int, x.shape)，断开跟踪。

2. 对于reshape、view操作时，-1的指定请放到batch维度。其他维度计算出来即可。batch维度禁止指定为大于-1的明确数字。如果是一维，那么直接指定为-1就好。


### 3.2 TensorRT 引擎生成
在`main_export.py`中添加生成engine的代码，此处遇到如下问题：
- 在设置网络精度为FP16的情况下，如何在不破坏tensorRT融合结果的情况下单独设置某层的精度为FP32? 

设置stict type并设置layer上的精度可以把它设置成fp16或fp32，可参考这个[例子](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/09-Advance/StrictType/strictType.py)（设置成fp16）
- 使用polygraphy找到第一个精度不符合要求的层后，如何继续去寻找下一个精度不符合要求的层？直接切割子图进行分块输出还是使用对修改第一个不符合要求层后生成的engine进行继续调试，命令诸如
    ```shell
    polygraphy run decoder.plan 
      --model-type engine 
      --trt 
      --load-inputs=custom_decoder_inputs.json 
      --trt-outputs "788"
      --fp16  
      --save-results=_fp16.json
    ```
我们根据导师的建议所采用的方法是：在网络层数不是非常深的情况下，逐个单改Transformer网络层中的某一层从fp16到fp32，看精度提高幅度，取提高幅度最高的，作为改变的第一层；然后迭代这一过程，直到精度合格。


### 3.3 性能热点分析及初步优化
使用trtexec命令分析当前模型各层执行时间，命令如下：

```shell
trtexec \
    --onnx=./mobilevit_s.onnx \
    --workspace=102400 \
    --minShapes=input:1x3x256x256 \
    --optShapes=input:16x3x256x256 \
    --maxShapes=input:64x3x256x256 \
    --dumpProfile=enable \
    > dumpOnlyAtten.txt
```
结果如下：

性能热点主要是Myelin所融合的3个Transformer block部分，分别耗时xxx。

随后编写LayerNorm、attention插件替换ONNX图中节点，测试后发现速度变慢，使用trt进行分析，以attention为里，trtexec执行结果如下所示：

结果表示我们所替换的插件破坏了Myelin对于Transformer block部分的融合，所以初步优化以失败告终

### 3.4 性能热点分析及进一步优化
通过上一部分分析发现TensorRT8.4已经对模型中Transformer block部分进行了比较高效的融合，如果想超过该结果需要进行更深层次的优化，所以决定开始编写基于FasterTransformer的插件，在编写插件过程中遇到了如下问题

#### 3.4.1 **变量生命周期问题**

TensorRT C++ api中`PluginField`的定义如下

```c++

class PluginField
{
public:
    AsciiChar const* name;
    void const* data;
    PluginFieldType type;
    int32_t length;
 
    PluginField(AsciiChar const* const name_ = nullptr, void const* const data_ = nullptr,
        PluginFieldType const type_ = PluginFieldType::kUNKNOWN, int32_t const length_ = 0) noexcept
        : name(name_)
        , data(data_)
        , type(type_)
        , length(length_)
    {
    }
};

```

通过上面我们可以看到，`PluginField`中的指针变量采用的是浅拷贝的方式，因此如果想正常使用就需要保证指针指向的变量的生命周期大于等于使用周期。


如下面代码，如果将`data_npz`定义为循环里面的局部变量，那么循环结束后`data_npz`占用内存就会被释放，从而导致`fieldCollections`中新增的`PluginField`的指针指向一块未被申请的内存，导致结果不正确。

```c++

std::vector<PluginField> fieldCollections;
std::string npz_name("npz_path");

cnpy::npz_t data_npz;

int npz_path_len = 0;
std::string npz_path_len_name("npz_path_len");
for(int i = 0; i < fc->nbFields; i++) {
    if(npz_path_len_name.compare(fc->fields[i].name) == 0) {
        npz_path_len = *((const int*)fc->fields[i].data);
    }
}

for(int i = 0; i < fc->nbFields; i++) {
    if(npz_name.compare(fc->fields[i].name) == 0) {
        string npz_path((const char*)fc->fields[i].data, npz_path_len);

        // cnpy::npz_t data_npz = cnpy::npz_load(npz_path);
        data_npz = cnpy::npz_load(npz_path);
        for(auto& iter: data_npz) {
            auto& item = iter.second;
            if(0 == type_id) {
                fieldCollections.emplace_back(PluginField(iter.first.c_str(), item.data<float>(), PluginFieldType::kFLOAT32, item.num_vals));
            } else if(1 == type_id) {
                fieldCollections.emplace_back(PluginField(iter.first.c_str(), item.data<half>(), PluginFieldType::kFLOAT16, item.num_vals));
            }
        }
    } else {
        fieldCollections.emplace_back(fc->fields[i]);
    }
}

```

#### 3.4.2 **变长属性的处理** 

`PluginField`在解析时候是需要指定长度的，但是对于一些可变长度数据来说这个事先是无法确定的。比如onnx中的字符串属性，在传入到TensorRT时，有没有`'\0'`结尾是不能保证的，因此需要设置一个属性来记录字符串的长度，这也就是上面`npz_path_len`的用处，对于其他比如float数组也是相同处理方式。

在mobilevit插件编写中，transformer的权重传入有三种方式

1. 当作输入传进去
2. 作为属性传入，由于属性的数量会随着encoder的层数变化，所以这里需要采用最大属性数量。此外还需要每个传入的数组数据额外添加一个长度属性。
3. 把权重npz路径当作属性传入，插件里解析npz来填充权重值

综合考虑源码的改动程度，我们采用第三种方案。


#### 3.4.3 plugin编译release和debug的问题

在debug模式下，plugin可以编译成功，但是在构建`engine`的时候会报`LLVM: out of memory`的错误。

在release模式下，plugin可以正常编译且正常构建`engine`。









<!-- 1. 负优化的心路历程 nsys-ui
2. mobileVit layer 继承自BaseLayer，仿照Vit
3. plugin 需要进行修改，确定输入输出格式，以及对plugin的测试  先达到FP16 -->