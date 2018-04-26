#include <iostream>
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

// function 1 (host & gpu versions)
// can replace with any other function
float func_1a( float input )
{
    return 1/(1+input*input);
}
__device__ float func_1b( float input )
{
    return 1/(1+input*input);
}

// function 2 (host & gpu versions)
float func_2a( float input )
{
    return ((1.0*input*input) + (3.0*input*input) + 5.0);
}
__device__ float func_2b( float input )
{
    return ((1.0*input*input) + (3.0*input*input) + 5.0);
}

//function 3 (host & gpu versions)
float func_3a( float input )
{
    return ((2.0*input*input*input) / (5.0*input*input));
}
__device__ float func_3b( float input )
{
    return ((2.0*input*input*input) / (5.0*input*input));
}

float trapezoidal( float a, float b, float n )
{
    float delta = (b-a)/n;
    float s = func_2a(a) + func_2a(b);
    for( int i = 1; i < n; i++ )
    {
        s += 2.0*func_2a(a+i*delta);
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
        d_output[tid] = func_2b(s) + func_2b(s + delta);
    }
}

int main()
{
    cudaFree(0);    // starts CUDA context, absorbs cost of startup


    // initializations
    cudaError_t err;
    float a = 0.0f;     // interval start
    float b = 1.0f;     // interval end
    int n = 10000;      // number of trapezoids
    float delta = (b-a)/n;
    float parallel_result = 0.0f;
    float* h_kernel_output = (float*)malloc(n * sizeof(float));
    float* d_kernel_output;
    cout.precision(5);

    // print out host function result
    cout << "Function 1: " << endl;
    cout << "Serial: Value of integral is " << trapezoidal(a, b, n) << endl;
    

    /* 
        Now the parallel part.
        The cudaMalloc was taking tons of time when I tested, not sure why.
        That's why I made the cudaFree(0) at the beginning.
        It absorbs the time cost of setting up the CUDA context,
            so the cudaMalloc() then takes relatively little time.
    */
    err = cudaMalloc( (void**) &d_kernel_output, n * sizeof(float) );
    gpu_err_chk(err);
    err = cudaMemcpy( d_kernel_output, h_kernel_output, n * sizeof(float), cudaMemcpyHostToDevice );
    gpu_err_chk(err);
    
    
    // call kernel function
    dim3 dimGrid (40);      // threads/n -> 256 threads/block -> 40 blocks needed
    dim3 dimBlock (256); // 256
    trapezoidal_kernel<<<dimGrid, dimBlock>>>( a, b, n, d_kernel_output);
    err = cudaGetLastError();
    gpu_err_chk(err);
    
    
    // copy data back from device
    err = cudaMemcpy( h_kernel_output, d_kernel_output, n * sizeof(float), cudaMemcpyDeviceToHost );
    gpu_err_chk(err);

    
    // get correct sum of trapezoid array
    for( int i=0; i<n; i++ )
    {
        parallel_result += h_kernel_output[i];
    }
    parallel_result *= delta/2.0;

    
    // print out device function result
    printf("Parallel: Value of integral is %6.4f\n", parallel_result);


    // free up memory
    free(h_kernel_output);
    cudaFree(d_kernel_output);
    
    return 0;
}