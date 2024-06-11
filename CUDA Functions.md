### cudaMalloc( (void\*\*)&dev_var, sizeof(int) )
第一个参数是指针，指向用于保存新分配的内存地址的变量，这里采用`void**`的原因是因为要先指向主机地址再指向设备地址。
第二个参数是分配内存的大小。
<font color = orange>注意:</font>不能在主机代码中对`cudaMalloc()`返回的指针进行解引用(Dereference)。
主机代码中可以把这个指针作为参数传递，可以做算术运算，但绝不可以用于读写内存。
总结如下：
- You can: 把`cudaMalloc()`分配的指针传递给在设备上执行的函数
- You can: 在设备代码中，对`cudaMalloc()`分配的指针进行内存读写操作
- You can: 把`cudaMalloc()`分配的指针传递给主机上执行的函数
- Never: 在主机代码中，对`cudaMalloc()`分配的指针进行内存读写
> [关于Reference与Deference](https://stackoverflow.com/questions/14224831/meaning-of-referencing-and-dereferencing-in-c)

> **Referencing** means taking the address of an existing variable (using &) to set a pointer variable. In order to be valid, a pointer has to be set to the address of a variable of the same type as the pointer, without the asterisk(*)：
```
int c1;
int *p1;
c1 = 6;
p = &c1;
```
> **Dereferencing** a pointer means using the * operator (asterisk character) to retrieve the value from the memory address that is pointed by the pointer: NOTE: The value stored at the address of the pointer must be a value OF THE SAME TYPE as the type of variable the pointer "points" to, but there is **no guarantee** this is the case unless the pointer was set correctly. The type of variable the pointer points to is the type less the outermost asterisk.
```
int d1;
d1 = *p1;
```
---
### cudaFree(dev_var)
类似于C中的`Free()`，用于释放`cudaMalloc()`分配的内存。

---
### cudaMemcpy(&var, dev_var, sizeof(int), cudaMemcpyDeviceToHost)
源指针为`dev_var`，目标指针为`&var`
`cudaMemcpyDeviceToHost`参数是用于说明，源指针是一个设备指针，而目标指针是一个主机指针，此时完成了主机对设备内存的访问。
如果设备想要访问主机内存，则要通过`cudaMemcpy(dev_var, &var, sizeof(int), cudaMemcpyHostToDevice)`。
如果两个指针都位于设备上，则使用`cudaMemcpyDeviceToDevice`。
如果两个指针都位于主机上，直接使用C标准的`memcpy()`函数即可。

---
### cudaGetDeviceCount
使用
```
cudaDeviceProp prop;

int count;
cudaGetDeviceCount(&count);

for(int i = 0; i < count; i++)
{
    cudaGetDeviceProperties(&prop, i);
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Device copy overlap: ");
    if(prop.deviceOverlap)
        printf("Enabled\n");
    else
        printf("Disabled\n");
}
```
可以获取设备的属性信息。

---
### cudaChooseDevice&cudaSetDevice
`cudaChooseDevice`用于寻找符合特定条件的设备，`cudaSetDevice`用于指定操作在特定设备上执行。例如要选择一个`Compute capability`版本为8.3的设备：
```
cudaDeviceProp prop;
int dev;

cudaGetDevice(&dev);
memset(&prop, 0, sizeof(cudaDeviceProp));
prop.major = 8;
prop.minor = 3;
cudaChooseDevice(&dev, &prop);
cudaSetDevice(dev);
```
 ---
 ### 核函数参数
 `kernel<<<2, 1>>>`表示运行时将创建核函数的两个副本，以并行方式运行，每个并行运行环境称为一个线程块(Block)。
 通过`blockIdx`来指定每个线程块的索引，
 ```
 __global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if (tid < N)    //防止非法访存
    {
        c[tid] = a[tid] + b[tid];
    }
}
 ```
 所有的线程块集合称为一个线程格(Grid)。
 更加详细的来说，`kernel<<<Dg, Db, Ns, S>>>`共有四个参数设置
 - `Dg`指的是Grid的维度，即一个Grid内包含了多少个Block，每个核函数只有一个Grid。Grid是Dim3类型，具体来说就是Dim3(Dg.x, Dg.y, 1)，第一维表示每行有多少个Block，第二维表示每列有多少个Block，第三维默认为1。由于硬件限制，Dg.x和Dg.y数量不能超过65535。
 - `Db`指的是Block的维度，即每个Block内有多少个线程，同样为Dim3类型，Dim3(Db.x, Db.y, Db.z)，这里的数值限制与`compute capability`有关。
 - `Ns`表示每个Block除了静态分配的Shared Memory之外还能够动态分配多少的Shared Memory，单位为`byte`，缺省值为0。
 - `S`为`cudaStream_t`类型参数，表示核函数位于哪个流里面，默认为0。
