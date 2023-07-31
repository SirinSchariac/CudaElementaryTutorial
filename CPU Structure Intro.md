## CPU Structure Intro

程序平均意义上的所需时间：CPI*时钟周期

 取指IF$\rightarrow$译码ID $\rightarrow$执行EXE $\rightarrow$访存MEM $\rightarrow$写回WB

> 流水线

指令级并行：能够极大的减少时钟周期，但是延迟和芯片面积会增加；依赖关系和分支的处理是一个问题。

旁路Bypassing：其实就是数据的Forwarding，解决数据相关。

停滞Stall

分支预测Branch Prediction 

超标量Superscalar：每个时钟周期发射多条指令

> 缓存

空间和时间的局部性Locality

L1 Cache $\rightarrow$ L2 Cache $\rightarrow$  L3 Cache $\rightarrow$  Main Memory $\rightarrow$  Disk

从左到右，存储空间增加，读写速度下降

> CPU内部并行性

- 指令级并行
  - 超标量
  - 乱序执行

- 数据级并行
  - 矢量计算

- 线程级并行
  - 同步多线程
  - 多核

 