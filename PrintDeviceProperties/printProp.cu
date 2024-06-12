#include <cstdio>

int main(void)
{
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);

    for(int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);

        printf("    --- General Information For Device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Device copy overlap:    ");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execution timeout:   ");
        if(prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Max Thread Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
    }
    
    return 0;

}