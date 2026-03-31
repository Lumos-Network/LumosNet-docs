<img src="../img/Lumos.png"/>

# 快速入门

Lumos允许您快速实现深度学习模型，我们提供了简洁的接口，您可以在lumos/include目录下的lumos.h文件中查看我们提供的接口

下面我们将用Lenet5模型实现MNIST手写数字识别，让您快速了解Lumos框架的使用，完整的框架教程，请您参考Lumos教程



### 模型构建

我们通常将一个深度学习模型视为一个计算图，所以在Lumos中一个深度学习模型就是一个计算图，我们需要首先创建一个计算图类的实例

```c
Graph *g = create_graph()
```

在此之后您需要创建不同的计算层，并确定它们的链接方式

```c
Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
Layer *l2 = make_avgpool_layer(2, 2, 0);
Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
Layer *l4 = make_avgpool_layer(2, 2, 0);
Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
Layer *l6 = make_im2col_layer();
Layer *l7 = make_connect_layer(84, 1, "relu");
Layer *l8 = make_connect_layer(10, 1, "relu");
Layer *l9 = make_softmax_layer(10);
Layer *l10 = make_mse_layer(10);
```

```c
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
```

append_layer2grpah将您创建的计算层按顺序添加到计算图中，此时我们创建的计算图g，就是一个完整的静态深度学习模型

Lumos默认对模型采用Kaiming初始化方案进行参数初始化，您也可以指定您希望的初始化方案，详细用法请参考后续的完整教程

完成模型创建后，我们需要调度模型进行计算，Lumos提供Session会话类来完成全部的计算调度，首先我们需要实例化一个会话

```c
Session *sess = create_session(g, 32, 32, 1, 10, type, path);
```

此处type为一个字符串（"CPU"，"GPU"）指定运行设备，path为一个字符串指定加载权重文件，如果您希望从零开始训练您的模型而不是迁移学习，那么请将path置为NULL

如下设置训练超参数

```c
set_train_params(sess, 10, 16, 16, 0.001);
```

Lumos提供了丰富的优化器选择，以SGDOptimizer为例

```c
SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
```

在训练开始前，Lumos需要完成内存等训练环境初始化

```c
init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
```

Lumos从您提供的train.txt和train_label.txt中获取训练数据的路径和真实标签的路径，其组织形式将在后续进行讲解

现在一切准备就绪，可以开始训练了

```c
train(sess);
```



### 数据组织

您需要提供数据路径让Lumos获取您的数据集，您需要提供两个文本文件“train.txt    label.txt”，其内容所示如下

train.txt中保存所有训练数据的路径

```
../../name.png
...
...
```

label.txt中保存所有数据标签的路径

```
../../name.txt
...
...
```

train.txt和label.txt中的每一行路径都是一一对应关系

您可以查看Lumos/demo中的xor数据集实例帮助您理解数据组织形式



### 编译与运行

现在您可以开始编译并运行您的模型了，首先您模型代码的实现应该在Lumos/demo中的lumos.c文件中，当然这不是必须的，您事实上可以将您的代码放在任何地方，但是Lumos提供的编译脚本默认将Lumos/demo/lumos.c作为编译对象，所以在您不了解编译脚本机制时，不妨将您的代码放在lumos.c文件中，以防不必要的麻烦

现在您只需要运行Lumos提供的编译脚本即可

```bash
make clean
make
```

编译过程最终形成一个可运行程序lumos.exe，请直接运行它

```bash
./lumos.exe
```