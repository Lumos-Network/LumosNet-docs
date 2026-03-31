<img src="../img/Lumos.png"/>

# Lumos安装

首先您需要从Lumos开源仓库将代码复制到您的设备上，您可以直接下载压缩包，也可使用git工具

```bash
git clone git@github.com:Lumos-Network/LumosNetwork.git
```

请确保您已安装完成编译器、make工具以及CUDA Toolkit，本文档在此不再赘述相关安装配置问题，如遇任何问题请参考相关工具的官方文档

请您进入Lumos项目的根目录，根目录下的makefile为编译脚本，其中CUDA Tooilkit路径索引按需修改为您的安装路径

```bash
COMMON+= -DGPU -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
```

Lumos中的demo目录下实现了多个经典网络模型实例，其中lumos.c文件为运行主文件，我们为您提供了一个简单的测试案例，请在lumos.c文件中添加如下代码

```c
#include "xor.h"

int main()
{
    xor_detect("gpu", "./demo/xor.lw");
}
```

现在进入命令行，执行以下命令

```bash
make clean
make
```

请等待编译结束，然后执行如下命令

```bash
./lumos.exe
```

如果一切顺利您将得到如下结果

```
[Lumos]         Module Structure
Connect         Layer    :    [output=   8, bias=1, active=relu]
Connect         Layer    :    [output=  16, bias=1, active=relu]
Connect         Layer    :    [output=   2, bias=1, active=linear]
CrossEntropy    Layer    :    [output=   1]

Get Train Data List From ./data/xor/data.txt

Get Label List From ./data/xor/label.txt

Start To Init Graph
[Lumos]                     Inputs         Outputs
Connect         Layer      1*  1*  2 ==>   1*  1*  8
Connect         Layer      1*  1*  8 ==>   1*  1* 16
Connect         Layer      1*  1* 16 ==>   1*  1*  2
CrossEntropy    Layer      1*  1*  2 ==>   1*  1*  1

Get Train Data List From ./backup/train.txt

Session Start To Running
./backup/data/0
Truth     Detect
1.000 0.904
0.000 0.096
Loss:0.0000

./backup/data/1
Truth     Detect
0.000 0.011
1.000 0.989
Loss:0.0000

./backup/data/2
Truth     Detect
1.000 0.959
0.000 0.041
Loss:0.0000

./backup/data/3
Truth     Detect
0.000 0.018
1.000 0.982
Loss:0.0000

Detct Classification: 4/4  1.00
```

您已完成Lumos的安装，并已实现了一个DNN网络的运行，您的Lumos旅程从此开始