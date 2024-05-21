# PytorchBertBiLstmClassify
Pytorch Bert+BiLstm二分类

LSTM的输出主要包括三个部分：output、h_n、c_n。1

output：这是LSTM网络在每个time step的输出，它是一个三维张量，其中第一维表示序列长度，第二维表示一批的样本数（batch），第三维是隐藏层大小（hidden_size）乘以方向数（num_directions）。如果LSTM是双向的，那么output将包含每个time step的正向和逆向的输出，连接在一起。

h_n：这个输出保存了每一层在最后一个time step的输出h。如果LSTM是双向的，h_n将分别保存前向和后向的最后一个time step的输出h。h_n是一个三维张量，其第一维是层数乘以方向数（num_layers*num_directions），第二维是batch size，第三维是隐藏层的大小。

c_n：与h_n的结构一致，但它保存的是c的值。c代表LSTM的cell state，用于存储LSTM的记忆单元。

例如，对于一个双向LSTM，如果我们定义了3个层，那么h_n的第一个维度的大小将是6（2*3），表示第一层前向传播和后向传播最后一个time step的输出。同样，c_n的结构与h_n相同，也保存了这些值。

这些输出提供了LSTM网络在处理序列数据时的关键信息，包括每个time step的输出、每层在序列结束时的状态以及cell state的值。这些信息对于后续的预测或分类任务非常重要。

参考：https://blog.csdn.net/qq_45812502/article/details/127297696

基于Pytorch Bert+BiLstm二分类最终抽取的其实是最后一个时刻的状态和参数，用以二分类。
