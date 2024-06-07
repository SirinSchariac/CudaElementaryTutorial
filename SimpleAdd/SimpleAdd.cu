#include <iostream>

__global__ void AddKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int len = 5;
    const int a[len] = {1, 2, 3, 4, 5};
    const int b[len] = {11, 22, 33, 44, 55};

    int c[len] = {0};

    int *dev_a, *dev_b, *dev_c;

    //allocate memory on GPU
    cudaMalloc((void**)&dev_a, len * sizeof(int));
    cudaMalloc((void**)&dev_b, len * sizeof(int));
    cudaMalloc((void**)&dev_c, len * sizeof(int));

    cudaMemcpy(dev_a, a, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, len*sizeof(int), cudaMemcpyHostToDevice);

    AddKernel<<<1, len>>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, len*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for(int i = 0;i < len; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}