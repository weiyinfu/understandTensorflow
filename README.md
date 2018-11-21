了解一件事物最好的方式就是实现一遍。  
tensorflow看上去复杂，是因为tensorflow包含了太多可有可无的功能，是过度设计了。它最初的理想并没有这么复杂。  

# TODO
* 神经网络初始化非常重要，好的初始化是成功的一半。看似简简单单，实际上有的初始化能够收敛，有的初始化就无法收敛。
* 输入形状为None的placeholder
* 支持更多形状的张量运算（目前一些运算支持的张量形状有限）
* 模型保存和加载   
只需要保存variable类型的Num，其余的没必要保存  
考虑模型保存和加载，必定要考虑序列化和反序列化
* layers   
全连接、卷积、RNN层的封装
* 模型级的封装
keras中有model，tensorflow中有estimator
* 使用cuda自定义worker来加速计算，或者使用C++重写worker，实现worker只需要实现所有Op子类的forward操作，不需要实现backward操作，构图过程都在python进行就可以，worker只负责运行图
