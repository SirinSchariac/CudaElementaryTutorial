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
