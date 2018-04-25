#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// function
// can replace with any other function
float func( float input )
{
    return 1/(1+input*input);
}

float trapezoidal( float a, float b, float n )
{
    float h = (b-a)/n;
    float s = func(a) + func(b);
    for( int i = 1; i < n; i++ )
    {
        s += 2*func(a+i*h);
    }
    return (h/2)*s;
}

__global__ void trapezoidal_kernel( float a, float b, float n, float (*f)(float), float* output )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float h = (b-a)/n;
    float s = f(a) + f(b);

    if( tid < n )   // might get messed up because of multiple blocks?
    {
        s+= 2*f(a+tid*h);
    }
    *output = (h/2)*s;
}

int main()
{
    cudaError_t err;
    float x0 = 0;
    float xn = 1;
    int n = 6;
    float* kernel_output;

    printf("Serial: Value of integral is %6.4f\n", 
        trapezoidal(x0, xn, n));
    
    // now the parallel part.
    err = cudaMalloc(  );   // for kernel_output
    gpu_err_chk(err);       // create this function above
    err = cudaMemcpy( , , cudaMemcpyHostToDevice );   // for kernel_output
    gpu_err_chk(err);
    
    dim3 dimGrid (32);
    dim3 dimBlock (16, 16);
    *kernel_output = 0;
    trapezoidal_kernel<<<dimGrid, dimBlock>>>( x0, xn, n, func, kernel_output);
    
    err = cudaMemcpy( , , cudaMemcpyDeviceToHost );
    gpu_err_chk(err);
    
    return 0;
}
