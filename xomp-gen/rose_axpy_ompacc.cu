// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
/* change this to do saxpy or daxpy : single precision or float precision*/
#define REAL float
#define VEC_LEN 1024000 //use a fixed number for now
/* zero out the entire vector */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

void zero(float *A,int n)
{
  int i;
  for (i = 0; i < n; i++) {
    A[i] = 0.0;
  }
}
/* initialize a vector with random floating point numbers */

void init(float *A,int n)
{
  int i;
  for (i = 0; i < n; i++) {
    A[i] = ((float )(drand48()));
  }
}
/*serial version */

void axpy(float *x,float *y,long n,float a)
{
  int i;
  for (i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}
/* compare two arrays and return percentage of difference */

float check(float *A,float *B,int n)
{
  int i;
  float diffsum = 0.0;
  float sum = 0.0;
  for (i = 0; i < n; i++) {
    diffsum += fabs(A[i] - B[i]);
    sum += fabs(B[i]);
  }
  return diffsum / sum;
}

__global__ void OUT__1__10503__(int n,float a,float *_dev_x,float *_dev_y)
{
  int _p_i;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(0,n - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,n - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
      _dev_y[_p_i - 0] += a * _dev_x[_p_i - 0];
    }
}

void axpy_ompacc(float *x,float *y,int n,float a)
{
  int i;
  /* this one defines both the target device name and data environment to map to,
     I think here we need mechanism to tell the compiler the device type (could be multiple) so that compiler can generate the codes of different versions; 
     we also need to let the runtime know what the target device is so the runtime will chose the right function to call if the code are generated 
#pragma omp target device (gpu0) map(x, y) 
   */
  {
    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    xomp_deviceDataEnvironmentEnter(0);
    float *_dev_x;
    int _dev_x_size[1] = {n};
    int _dev_x_offset[1] = {0};
    int _dev_x_Dim[1] = {n};
    _dev_x = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)x,1,sizeof(float ),_dev_x_size,_dev_x_offset,_dev_x_Dim,1,0)));
    float *_dev_y;
    int _dev_y_size[1] = {n};
    int _dev_y_offset[1] = {0};
    int _dev_y_Dim[1] = {n};
    _dev_y = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)y,1,sizeof(float ),_dev_y_size,_dev_y_offset,_dev_y_Dim,1,1)));
    /* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
    int _num_blocks_ = xomp_get_max1DBlock(0,n - 1 - 0 + 1);
    OUT__1__10503__<<<_num_blocks_,_threads_per_block_>>>(n,a,_dev_x,_dev_y);
    xomp_deviceDataEnvironmentExit(0);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cumulative_time = cumulative_time + time;
    printf ("1-D thread block number =%d threads per block =%d\n",_num_blocks_, _threads_per_block_ );
    printf("data alloc+ transfer + Kernel launch +exe time:  %3.5f ms \n", cumulative_time);
  }
}

int main(int argc,char *argv[])
{
  xomp_acc_init();
  int n;
  float *y_ompacc;
  float *y;
  float *x;
  float a = 123.456;
  n = 1024000;
  printf ("vec length  =%d \n", n);
  y_ompacc = ((float *)(malloc(n * sizeof(float ))));
  y = ((float *)(malloc(n * sizeof(float ))));
  x = ((float *)(malloc(n * sizeof(float ))));
  srand48((1 << 12));
  init(x,n);
  init(y_ompacc,n);
  memcpy(y,y_ompacc,n * sizeof(float ));
  axpy(x,y,n,a);
/* openmp acc version */
  axpy_ompacc(x,y_ompacc,n,a);
  float checkresult = check(y_ompacc,y,n);
  printf("axpy(%d): checksum: %g\n",n,checkresult);
  assert(checkresult < 1.0e-7);
  free(y_ompacc);
  free(y);
  free(x);
  return 0;
}
