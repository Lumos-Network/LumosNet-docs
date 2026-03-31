<img src="../img/Lumos.png"/>

# XOR

### 异或（XOR）问题

异或函数XOR，是两个二进制数a，b的运算，当且仅当其中一个值为1时，XOR结果为1，其余结果为0

**异或**

| 标签 | 数据   |
| ---- | ------ |
| 1    | [1, 0] |
| 1    | [0, 1] |
| 0    | [1, 1] |
| 0    | [0, 0] |

异或问题是典型的非线性问题

**逻辑与**

| 标签 | 数据   |
| ---- | ------ |
| 1    | [1, 1] |
| 0    | [1, 0] |
| 0    | [0, 1] |
| 0    | [0, 0] |

**逻辑或**

| 标签 | 数据   |
| ---- | ------ |
| 1    | [1, 0] |
| 1    | [0, 1] |
| 1    | [1, 1] |
| 0    | [0, 0] |

异或，逻辑与，逻辑或的散点图如下

<img src="../img/异或.png"/>

可以看出，逻辑与和逻辑或的数据分布可以用一个线性函数进行分割，而异或无法用单一线性函数进行划分，所以XOR具有典型非线性



### 数据集

Lumos框架已提供xor数据集，在Lumos项目demo目录下



### 代码构建

我们构建一个简单的全连接神经网络来解决XOR问题，其网络结构如下

<img src="../img/异或_1.png"/>

使用Lumos框架构建网络模型

```c
Graph *g = create_graph();
Layer *l1 = make_connect_layer(8, 1, "relu");
Layer *l2 = make_connect_layer(16, 1, "relu");
Layer *l3 = make_connect_layer(2, 1, "linear");
Layer *l4 = make_crossentropy_layer(2);
```

我们使用crossentropy分类器进行分类

接下来构建会话，并设置相关训练超参数

```c
Session *sess = create_session(g, 1, 2, 1, 2, type, path);
set_train_params(sess, 50, 4, 4, 0.1);
SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
init_session(sess, "./demo/xor/data.txt", "./demo/xor/label.txt");
```

我们使用SGD参数优化器进行参数优化

完整代码如下

```c
#include "xor.h"

void xor(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(8, 1, "relu");
    Layer *l2 = make_connect_layer(16, 1, "relu");
    Layer *l3 = make_connect_layer(2, 1, "linear");
    Layer *l4 = make_crossentropy_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_train_params(sess, 50, 4, 4, 0.1);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./demo/xor/data.txt", "./demo/xor/label.txt");
    train(sess);
}

void xor_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(8, 1, "relu");
    Layer *l2 = make_connect_layer(16, 1, "relu");
    Layer *l3 = make_connect_layer(2, 1, "linear");
    Layer *l4 = make_crossentropy_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./demo/xor/data.txt", "./demo/xor/label.txt");
    detect_classification(sess);
}
```

在Lumos框架中demo目录下，您能找到xor.c文件，这就是我们已实现的XOR模型



### 结果展示

<img src="../img/xor_loss.png"/>

该网络在经过50个epoch训练后，可以准确的对XOR数据进行分类，分类精度100%