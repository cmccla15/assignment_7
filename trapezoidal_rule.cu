#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

float function( float input )
{
    return 1/(1+input*input);
}

float trapezoidal( float a, float b, float n )
{
    float h = (b-a)/n;
    float s = function(a) + function(b);
    for( int i = 1; i < n; i++ )
    {
        s += 2*function(a+i*h);
    }
    return (h/2)*s;
}

__global__ trapezoidal_kernel( float a, float b, float n, float (*f)(float, float, float) )
{
    int tid = threadIdx.x;
}

int main()
{
    float x0 = 0;
    float xn = 1;
    int n = 6;

    printf("Serial: Value of integral is %6.4f\n", 
        trapezoidal(x0, xn, n));
    
    //now the parallel part.



    
    return 0;
}