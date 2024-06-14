### 并行线程块的分解
启动N个线程块，每个线程块内有1个线程：`kernel<<<N, 1>>>`。
启动1个线程块，线程块内有N个线程：`kernel<<<1, N>>>`。
对于线程块，硬件限制其数量不超过65535。
对于线程，其最大的数量与`maxThreadsPerBlock`有关，经过测试在本机环境下(NVIDIA GeForce RTX 3060 Laptop)，该数值为1024。

### 超长矢量的操作
如果矢量的长度超过了1024，就无法通过一个线程块内的并行线程完成操作。
对于这一问题，可以通过改动核函数的索引计算方式与调用方式来解决。
> **索引计算上**
```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```
类似于二维数组的计算方式，如图所示。

![Block and Thread](/Pic/BlockAndThread.png)

> **调用方式上**

对于一个大小为N的矢量，每个线程块内有1024个线程的情况下，需要使用$\frac{N}{1024}$个线程块，但由于`int`本身的向下取整问题，如果大小为1023，就会导致调用了0个线程块的情况。

为了避免这一问题，需要除法进行向上取整，此时可以通过$\frac{N+1023}{1024}$来解决这一问题。因此，核函数的调用变为：
```
kernel<<<(N+1023)/1024, 1024>>>(dev_a, dev_b, dev_c);
```
这种情况下会启动多余的线程，但之前所说的`if(tid < N)`的语句进行了检查，避免了越界。
> **如果矢量长度非常长，超过了65535*1024呢?**

这种情况下，按照上述的方式启动核函数会失败。
因此类似在CPU上的并行处理思想，采用`while`内递增的方式，将递增的步长设置为整个线程格中正在运行的线程数量。
```
__global__ void add(int *dev_a, int *dev_b, int *dev_c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while( tid < N )
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
```
此时对于核函数的调用，就不能再是`<<<(N+1023)/1024, 1024>>>`了。这里的参数只要在满足65535和`maxThreadsPerBlock`的情况下，额可以任意设置，一般来说可以设定为`<<<1024, 1024>>>`。

> **总结**

- 当矢量长度小于`maxThreadsPerBlock`的时候，启用一个线程块即可。
- 当矢量长度超过`maxThreadsPerBlock`而仍小于65535*`maxThreadsPerBlock`时，需要启动多个线程块(改动索引和调用)。
- 当矢量长度超过65535*`maxThreadsPerBlock`时，需要启动多个线程格(改动为递增索引)。