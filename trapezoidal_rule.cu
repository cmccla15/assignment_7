#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

inline void gpu_handle_error( cudaError_t err, const char* file, int line, int abort = 1 )
{
	if (err != cudaSuccess)
	{
		fprintf (stderr, "gpu error: %s, %s, %d\n", cudaGetErrorString (err), file, line);
		if (abort)
			exit (EXIT_FAILURE);
	}
}
#define gpu_err_chk(e) {gpu_handle_error( e, __FILE__, __LINE__ );}

// function (host & gpu versions)
// can replace with any other function
float func( float input )
{
    return 1/(1+input*input);
}
__device__ float func2( float input )
{
    return 1/(1+input*input);
}

float trapezoidal( float a, float b, float n )
{
    float delta = (b-a)/n;
    float s = func(a) + func(b);
    for( int i = 1; i < n; i++ )
    {
        s += 2.0*func(a+i*delta);
    }
    return (delta/2)*s;
}

__global__ void trapezoidal_kernel( float a, float b, float n, float* d_output )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float delta = (b-a)/n;
    float s = a + (float)tid * delta;

    if( tid < n )
    {
        d_output[tid] = func2(s) + func2(s + delta);
    }
}

int main()
{
    cudaError_t err;
    float a = 0.0f;        //interval start
    float b = 1.0f;        //interval end
    int n = 1000;      //number of trapezoids
    float parallel_result = 0.0f;
    float* h_kernel_output = (float*)malloc(n * sizeof(float));
    float* d_kernel_output;

    printf("Serial: Value of integral is %6.4f\n", trapezoidal(a, b, n));
    
    // now the parallel part.
    err = cudaMalloc( (void**) &d_kernel_output, n * sizeof(float) );
    gpu_err_chk(err);
    err = cudaMemcpy( d_kernel_output, h_kernel_output, n * sizeof(float), cudaMemcpyHostToDevice );
    gpu_err_chk(err);
    
    dim3 dimGrid (4);  // threads/n -> 256 threads/block -> 4 blocks needed
    dim3 dimBlock (16, 16);
    trapezoidal_kernel<<<dimGrid, dimBlock>>>( a, b, n, d_kernel_output);
    err = cudaGetLastError();
    gpu_err_chk(err);
    
    err = cudaMemcpy( h_kernel_output, d_kernel_output, n * sizeof(float), cudaMemcpyDeviceToHost );
    gpu_err_chk(err);

    for( int i=0; i<n; i++ )
    {
        parallel_result += h_kernel_output[i];
    }
    parallel_result *= 2.0*((b-a)/n);

    printf("Parallel: Value of integral is %6.4f\n", parallel_result);

    free(h_kernel_output);
    cudaFree(d_kernel_output);
    
    return 0;
}