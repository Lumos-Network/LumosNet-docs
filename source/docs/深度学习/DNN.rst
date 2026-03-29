DNN
-------------------------

深度全连接神经网络的基本过程是最清晰的。由于该模型可以更好的划分为线性模型的复合，所以其推到流程十分简单。

模型抽象
^^^^^^^^

设有一个DNN模型有以下参数设置

-  损失函数L
-  隐层激活函数σ
-  隐层权重w
-  隐层偏移量b
-  梯度δ

我们做如下声明

隐层计算输出（没有激活）z，隐层计算输出（通过激活）a

隐层计算如下：

.. math::


   \begin{aligned}
   &z_{l}=w_{l}a_{l-1}+b_{l} \\
   &a_{l}=\sigma(z_{l})=\sigma(w_{l}a_{l-1}+b_{l})
   \end{aligned}

设隐层计算为一个函数g，则整个DNN模型可表示为函数f

.. math::


   f=L(g_{l}(g_{l-1}(...g(x))))

可以看出整个DNN模型是一个线性函数的复合

反向推导
^^^^^^^^

由于我们利用梯度下降法来更新模型权重，所以我们需要求每一个权重关于损失的梯度

求梯度，我们就要求损失关于权重的偏导（梯度：全部偏导数组成的向量）

.. math::


   \frac{\partial L}{\partial w}\quad \frac{\partial L}{\partial b}

由复合函数求导的链式法则，有如下过程：

设求第k层权重w，b的梯度

.. math::


   \begin{aligned}
   &D_{w_{k}}=\frac{\partial L}{\partial z_{L}} \dot\ \frac{\partial z_{L}}{\partial z_{L-1}} \dot\
   \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots\ \frac{\partial z_{k+1}}{\partial z_{k}} \dot\ \frac{\partial z_{k}}{\partial w_{k}} \\
   &D_{b_{k}}=\frac{\partial L}{\partial z_{L}} \dot\ \frac{\partial z_{L}}{\partial z_{L-1}} \dot\
   \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots\ \frac{\partial z_{k+1}}{\partial z_{k}} \dot\ \frac{\partial z_{k}}{\partial b_{k}}
   \end{aligned}

可以看出这是一个可递归过程，设输出层为第l（小写L）层，则输出层权重梯度为

.. math::


   \begin{aligned}
   &D_{w_{l}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial w_{l}} \\
   &D_{b_{l}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial b_{l}}
   \end{aligned}

第l-1层权重梯度为

.. math::


   \begin{aligned}
   &D_{w_{l-1}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \dot\ \frac{\partial z_{l-1}}{\partial w_{l-1}} \\
   &D_{b_{l-1}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \dot\ \frac{\partial z_{l-1}}{\partial b_{l-1}}
   \end{aligned}

第l-2层权重梯度为

.. math::


   \begin{aligned}
   &D_{w_{l-2}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \dot\ \frac{\partial z_{l-1}}{\partial z_{l-2}} \dot\ \frac{\partial z_{l-2}}{\partial w_{l-2}} \\
   &D_{b_{l-2}}=\frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \dot\ \frac{\partial z_{l-1}}{\partial z_{l-2}} \dot\
   \frac{\partial z_{l-2}}{\partial b_{l-2}}
   \end{aligned}

我们可以提取出公式中递归部分

.. math::


   \frac{\partial L}{\partial z_{l}} \quad \frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \quad \frac{\partial L}{\partial z_{l}} \dot\ \frac{\partial z_{l}}{\partial z_{l-1}} \dot\ \frac{\partial z_{l-1}}{\partial z_{l-2}}

则设

.. math::


   \begin{aligned}
   \delta_{k}&=\frac{\partial L}{\partial z_{L}} \dot\ \frac{\partial z_{L}}{\partial z_{L-1}} \dot\
   \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots\ \frac{\partial z_{k+1}}{\partial z_{k}} \\
   &=\delta_{k+1} \dot\ \frac{\partial z_{k+1}}{\partial z_{k}}
   \end{aligned}

所以对于每一层只需要计算

.. math::


   \frac{\partial z_{k+1}}{\partial z_{k}} \quad \frac{\partial z_{k}}{\partial w_{k}} \quad \frac{\partial z_{k}}{\partial b_{k}}

则逐层递归就可以求出每一层权重梯度了，再通过梯度下降公式更新权重

计算
^^^^

我们现在来计算

.. math::


   \frac{\partial z_{k+1}}{\partial z_{k}} \quad \frac{\partial z_{k}}{\partial w_{k}} \quad \frac{\partial z_{k}}{\partial b_{k}}

由隐层计算公式可知

.. math::


   \begin{aligned}
   \frac{\partial z_{k+1}}{\partial z_{k}}&=
   \frac{\partial (w_{k+1}a_{k}+b_{k+1})}{\partial z_{k}} \\
   &=\frac{\partial (w_{k+1} \sigma(z_{k})+b_{k+1})}{\partial z_{k}} \\
   &=(w_{k+1})^{T} \odot \sigma^{'}(z_{k})
   \end{aligned}

则

.. math::


   \begin{aligned}
   \delta_{k}&=(w_{k+1})^{T} \delta_{k+1} \odot \sigma^{'}(z_{k})
   \end{aligned}

再计算权重的偏导

.. math::


   \begin{aligned}
   &\frac{\partial z_{k}}{\partial w_{k}}
   =\frac{\partial (w_{k}a_{k-1}+b_{k})}{\partial w_{k}}=(a_{k-1})^{T} \\
   &\frac{\partial z_{k}}{\partial b_{k}}
   =\frac{\partial (w_{k}a_{k-1}+b_{k})}{\partial b_{k}}=1
   \end{aligned}

所以综上所述

.. math::


   \begin{aligned}
   &D_{w_{k}}=\delta_{k} (a_{k-1})^{T} \\
   &D_{b_{k}}=\delta_{k}
   \end{aligned}

实现
----

在代码实现时，有一些细节需要注意

以上推到过程中，我们总是计算Zk（未激活计算结果）的偏导，如果这样，在代码实现时，相关数据信息需要从上一层获取，这样的设计显然不合理

既然神经网络是一个复合函数，那么链式求导公式（如下），是不是也可以换一种写法

.. math::


   \begin{aligned}
   &D_{w_{k}}=\frac{\partial L}{\partial z_{L}} \dot\ \frac{\partial z_{L}}{\partial z_{L-1}} \dot\
   \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots\ \frac{\partial z_{k+1}}{\partial z_{k}} \dot\ \frac{\partial z_{k}}{\partial w_{k}} \\
   &D_{b_{k}}=\frac{\partial L}{\partial z_{L}} \dot\ \frac{\partial z_{L}}{\partial z_{L-1}} \dot\
   \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots\ \frac{\partial z_{k+1}}{\partial z_{k}} \dot\ \frac{\partial z_{k}}{\partial b_{k}}
   \end{aligned}

新写法如下

.. math::


   \begin{aligned}
   &D_{w_{k}}=\frac{\partial L}{\partial a_{L}} \dot\ \frac{\partial a_{L}}{\partial a_{L-1}} \dot\
   \frac{\partial a_{L-1}}{\partial a_{L-2}} \cdots\ \frac{\partial a_{k+1}}{\partial a_{k}} \dot\ \frac{\partial a_{k}}{\partial w_{k}} \\
   &D_{b_{k}}=\frac{\partial L}{\partial a_{L}} \dot\ \frac{\partial a_{L}}{\partial a_{L-1}} \dot\
   \frac{\partial a_{L-1}}{\partial a_{L-2}} \cdots\ \frac{\partial a_{k+1}}{\partial a_{k}} \dot\ \frac{\partial a_{k}}{\partial b_{k}} \\
   &a_{k}=\sigma(z_{k})
   \end{aligned}

这样做的意义在于，每一层计算所需的数据全部由本层数据决定，这样就做到了每一层计算的独立

因为上一层的输入等于下一层的输出，设每一层的输入为d，则

.. math::


   d_{k}=a_{k-1}

那么其他推导公式做如下调整

.. math::


   \begin{aligned}
   \frac{\partial a_{k}}{\partial a_{k-1}}
   &=\frac{\partial \sigma(w_{k}a_{k-1}+b_{k})}{\partial a_{k-1}} \\
   &=(w_{k})^{T} \odot\ \sigma^{'}(w_{k}a_{k-1}+b_{k}) \\
   &= (w_{k})^{T} \odot\  \sigma^{'}(a_{k})
   \end{aligned}

则

.. math::


   \begin{aligned}
   \delta_{k}&=(w_{k})^{T} \delta_{k+1} \odot \sigma^{'}(a_{k})
   \end{aligned}

再计算权重的偏导

.. math::


   \begin{aligned}
   &\frac{\partial a_{k}}{\partial w_{k}}
   =\frac{\partial \sigma(w_{k}a_{k-1}+b_{k})}{\partial w_{k}}=(a_{k-1})^{T} \odot\  \sigma^{'}(w_{k}a_{k-1}+b_{k})
   = (d_{k})^{T} \odot\ \sigma^{'}(a_{k})\\
   &\frac{\partial a_{k}}{\partial b_{k}}
   =\frac{\partial \sigma(w_{k}a_{k-1}+b_{k})}{\partial b_{k}}=\sigma^{'}(w_{k}a_{k-1}+b_{k})=\sigma^{'}(a_{k})
   \end{aligned}

所以综上所述

.. math::


   \begin{aligned}
   &D_{w_{k}}= (\delta_{k-1} \odot\ \sigma^{'}(a_{k}))d_{k} \\
   &D_{b_{k}}=\delta_{k-1} \odot \sigma^{'}(a_{k})
   \end{aligned}
