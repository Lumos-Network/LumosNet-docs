<img src="../img/Lumos.png"/>

# LeNet5

LeNet-5 是深度学习史上最经典的卷积神经网络，由 Yann LeCun 在 1998 年提出，专门用于**手写数字识别（MNIST 数据集）**，是所有 CNN 的鼻祖

其基本结构如下：

<img src="../img/lenet_01.png"/>



### MNIST数据集

MNIST（Modified National Institute of Standards and Technology database，改进型美国国家标准与技术研究院数据库）是计算机视觉与机器学习领域最经典、最常用的基准数据集之一，由 Yann LeCun 等人于 1998 年发布，常被称为深度学习的 “Hello World”

原始数据来自 NIST 的手写数字集，但 NIST 训练集取自人口普查局员工、测试集取自高中生，分布不够合理；MNIST 对其做了重混、尺寸归一化（28×28 像素）、居中对齐、抗锯齿灰度处理，解决了原数据集的缺陷，样本来自 250 名不同书写者，保证了多样性

示例如下：

<img src="../img/0_0.png"/>

您可以在kaggle上寻找并下载数据集



### 代码构建

使用Lumos框架构建网络模型

```c
Graph *g = create_graph();
Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
Layer *l2 = make_maxpool_layer(2, 2, 0);
Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
Layer *l4 = make_maxpool_layer(2, 2, 0);
Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
Layer *l6 = make_connect_layer(84, 1, "relu");
Layer *l7 = make_connect_layer(10, 1, "linear");
Layer *l8 = make_crossentropy_layer(10);
```

我们使用crossentropy分类器进行分类

接下来构建会话，并设置相关训练超参数

```c
Session *sess = create_session(g, 32, 32, 1, 10, type, path);
set_train_params(sess, 10, 16, 16, 0.001);
SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
```

我们使用SGD参数优化器进行参数优化

完整代码如下

```c
#include "lenet5_mnist.h"

void lenet5_mnist(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
    Layer *l6 = make_connect_layer(84, 1, "relu");
    Layer *l7 = make_connect_layer(10, 1, "linear");
    Layer *l8 = make_crossentropy_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_train_params(sess, 10, 16, 16, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
    train(sess);
}

void lenet5_mnist_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
    Layer *l6 = make_connect_layer(84, 1, "relu");
    Layer *l7 = make_connect_layer(10, 1, "linear");
    Layer *l8 = make_crossentropy_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/mnist/test.txt", "./data/mnist/test_label.txt");
    detect_classification(sess);
}
```

在Lumos框架中demo目录下，您能找到lenet5_mnist.c文件，这就是我们已实现的lenet5模型



### 结果展示

<img src="../img/lenet5_loss.png"/>

该网络在经过10个epoch训练后，分类精度在95%以上
