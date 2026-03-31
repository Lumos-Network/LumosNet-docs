<img src="../img/Lumos.png"/>

# ALEXNET

AlexNet 是深度学习与计算机视觉领域的里程碑式卷积神经网络（CNN），由 Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 于 2012 年提出，在 ImageNet 大规模视觉识别挑战赛（ILSVRC 2012）中以Top-5 错误率 15.3%（远超第二名 26.2%）夺冠，正式开启了深度学习在计算机视觉领域的爆发式应用

- 定位：面向大规模图像分类的深度 CNN，专为 ImageNet（1000 类、百万级图像）设计
- 意义：首次证明深度 CNN + 大规模数据 + GPU 加速可在复杂视觉任务上取得碾压式效果，奠定现代 CNN 基础，后续 VGG、GoogLeNet、ResNet 等均受其启发，推动深度学习从实验室走向工业界，成为 AI 浪潮的关键起点

其网络结构如下：

<img src="../img/alexnet_1.png"/>

### 

### 创新与突破

ReLU 激活函数（革命性）

- 替代传统 Sigmoid/Tanh，解决梯度消失问题，训练速度提升数倍
- 公式：`ReLU(x)=max(0, x)`，单侧饱和，正区间梯度恒为 1

Dropout 正则化

- 在 FC6、FC7 中以 50% 概率随机失活神经元，强制模型学习鲁棒特征，显著抑制过拟合

重叠最大池化

- 池化核 3×3、步长 2，池化区域重叠，相比传统非重叠池化，减少信息丢失、提升特征鲁棒性

局部响应归一化（LRN）

- 对 ReLU 输出做邻近通道归一化，模拟生物神经元侧抑制，提升泛化；后续 VGG 等证明效果有限，已被batchnormalization取代



### Flower数据集

我们提供的示例运用一个小型花卉数据集，我们在此不对数据集做过多介绍，您可以使用您的任意数据集在该模型上进行训练，我们的实例仅仅是展示在Lumos框架下构建模型的过程



### 代码构建

使用Lumos框架构建网络模型

```c
Graph *g = create_graph();
Layer *l1 = make_convolutional_layer(96, 11, 4, 2, 1, "relu");
Layer *l2 = make_maxpool_layer(3, 2, 0);
Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, "relu");
Layer *l4 = make_maxpool_layer(3, 2, 0);
Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
Layer *l8 = make_maxpool_layer(3, 2, 0);
Layer *l9 = make_dropout_layer(0.5);
Layer *l10 = make_connect_layer(4096, 1, "relu");
Layer *l11 = make_dropout_layer(0.5);
Layer *l12 = make_connect_layer(4096, 1, "relu");
Layer *l13 = make_connect_layer(5, 1, "linear");
Layer *l14 = make_crossentropy_layer(5);
```

我们使用crossentropy分类器进行分类

接下来指定各计算层的参数初始化

```c
init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l5, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");

init_kaiming_uniform_bias(l1, "fan_in");
init_kaiming_uniform_bias(l3, "fan_in");
init_kaiming_uniform_bias(l5, "fan_in");
init_kaiming_uniform_bias(l6, "fan_in");
init_kaiming_uniform_bias(l7, "fan_in");

init_kaiming_uniform_kernel(l10, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l12, sqrt(5.0), "fan_in", "leaky_relu");
init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");

init_kaiming_uniform_bias(l10, "fan_in");
init_kaiming_uniform_bias(l12, "fan_in");
init_kaiming_uniform_bias(l13, "fan_in");
```

构建会话，并设置相关训练超参数

```c
Session *sess = create_session(g, 224, 224, 3, 5, type, path);
float *mean = calloc(3, sizeof(float));
float *std = calloc(3, sizeof(float));
mean[0] = 0.485;
mean[1] = 0.456;
mean[2] = 0.406;
std[0] = 0.229;
std[1] = 0.224;
std[2] = 0.225;
transform_normalize_sess(sess, mean, std);
transform_resize_sess(sess, 224, 224);
set_train_params(sess, 50, 32, 32, 0.001);
SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
```

可以看到我们对数据集进行了一定的预处理操作，首先对数据集进行归一化，归一化的分布来自于ImageNet数据集的先验计算结果，后续我们对数据集进行缩放，使其符合网络模型输入

我们使用SGD参数优化器进行参数优化

完整代码如下

```c
#include "alexnet_flower.h"

void alexnet_flower(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 2, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_dropout_layer(0.5);
    Layer *l10 = make_connect_layer(4096, 1, "relu");
    Layer *l11 = make_dropout_layer(0.5);
    Layer *l12 = make_connect_layer(4096, 1, "relu");
    Layer *l13 = make_connect_layer(5, 1, "linear");
    Layer *l14 = make_crossentropy_layer(5);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l5, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l1, "fan_in");
    init_kaiming_uniform_bias(l3, "fan_in");
    init_kaiming_uniform_bias(l5, "fan_in");
    init_kaiming_uniform_bias(l6, "fan_in");
    init_kaiming_uniform_bias(l7, "fan_in");

    init_kaiming_uniform_kernel(l10, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l12, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l10, "fan_in");
    init_kaiming_uniform_bias(l12, "fan_in");
    init_kaiming_uniform_bias(l13, "fan_in");

    Session *sess = create_session(g, 224, 224, 3, 5, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 224, 224);
    set_train_params(sess, 50, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void alexnet_flower_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 2, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    Layer *l9 = make_dropout_layer(0.5);
    Layer *l10 = make_connect_layer(4096, 1, "relu");
    Layer *l11 = make_dropout_layer(0.5);
    Layer *l12 = make_connect_layer(4096, 1, "relu");
    Layer *l13 = make_connect_layer(5, 1, "linear");
    Layer *l14 = make_crossentropy_layer(5);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    Session *sess = create_session(g, 224, 224, 3, 5, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 224, 224);
    set_detect_params(sess);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    detect_classification(sess);
}
```

在Lumos框架中demo目录下，您能找到alexnet_flower.c文件，这就是我们已实现的alexnet模型



### 结果展示

<img src="../img/alexnet_loss.png"/>

该网络在经过35个epoch训练后，分类精度在95%左右