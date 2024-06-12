#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000
#define threshold 1000

/**
 * @Ryuu_Mei
 * Julia Set calculation and visualization
 * From CUDA By Example
 * 2024/6/12
 */

struct cuComplex
{
    float r;//real part
    float i;//imaginary part

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2(void) 
    {
        return r*r + i*i;
    }

    __device__ cuComplex operator * (const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, r*a.i + i*a.r);
    }

    __device__ cuComplex operator + (const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    /* 计算复平面坐标 */
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for(int i = 0; i < 200; i++)
    {
        a = a*a + c;
        if(a.magnitude2() > threshold)
        {
            return 0;
        }
    }

    return 1;
}

__global__ void kernel( unsigned char *ptr )
{
    int x = blockIdx.x;//线程块索引
    int y = blockIdx.y;//线程块索引
    int offset = x + y * gridDim.x;//gridDim=(DIM, DIM)表示线程格的大小

    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

int main(void)
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc((void**)dev_bitmap, bitmap.image_size() );

    dim3 grid(DIM, DIM);//二维线程格，用于复平面的计算
    kernel<<<grid, 1>>>( dev_bitmap );

    cudaMemcpy( bitmap.get_ptr(), 
                dev_bitmap,
                bitmap.image_size(),
                cudaMemcpyDeviceToHost);
    
    bitmap.display_and_exit();

    cudaFree( dev_bitmap );

    return 0;
}