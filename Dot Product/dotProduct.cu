#include <cstdio>

#define sum_square(x) (x * (x+1) * (2*x+1) / 3)
#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0;
    while(tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;//total threads num in the grid
    }
    cache[cacheIdx] = temp;

    __syncthreads();//synchronize all the threads

    //production
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if(cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }
    if(cacheIdx == 0)
    {
        c[blockIdx.x] = cache[0];
    }
    
}

int main(void)
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));

    /*  Each block will have a copy of partial_c
        so the size is blocksPerGrid */
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    for(int i = 0;i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }
    
    cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    printf("Does GPU value %.6g = %.6g ? \n", c, sum_square((float)N));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
    
    return 0;
}