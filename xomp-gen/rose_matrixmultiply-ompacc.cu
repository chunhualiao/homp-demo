/*
Naive matrix-matrix multiplication(mmm)
By C. Liao
*/
#include <stdio.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define N 1024 
#define M 1024
#define K 1024
#define REAL float 
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 
int i;
int j;
int k;
float a[1024][1024];
float b[1024][1024];
float c[1024][1024];
float c2[1024][1024];
int init();
int mmm();
int mmm2();
int verify();

int main()
{
  xomp_acc_init();
  init();
  mmm();
  mmm2();
  return verify();
}

int init()
{
  for (i = 0; i < 1024; i++) 
    for (j = 0; j < 1024; j++) 
      a[i][j] = (3.0 * i * j / 1024 / 1024);
  for (i = 0; i < 1024; i++) 
    for (j = 0; j < 1024; j++) 
      b[i][j] = (5.0 * j * i / 1024 / 1024);
  for (i = 0; i < 1024; i++) 
    for (j = 0; j < 1024; j++) {
      c[i][j] = 0.0;
      c2[i][j] = 0.0;
    }
  return 0;
}
/*
TODO: try different i,j,k orders
a b     e f    a*e+ b*g , a*f+ b*h
c d  x  g h  = c*e+ d*g,  c*f+ d*h
*/

__global__ void OUT__1__11560__(float *_dev_a,float *_dev_b,float *_dev_c)
{
  int _p_i;
  int _p_j;
  int _p_k;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(0,1023,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,1023,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
      for (_p_j = 0; _p_j < 1024; _p_j++) 
        for (_p_k = 0; _p_k < 1024; _p_k++) 
          _dev_c[_p_i * 1024 + _p_j] = _dev_c[_p_i * 1024 + _p_j] + _dev_a[_p_i * 1024 + _p_k] * _dev_b[_p_k * 1024 + _p_j];
    }
}

int mmm()
{
//For static arrays with known dimension info. , no array section info. is needed  
//#pragma omp target map(tofrom:c[0:N][0:M]), map(to:a[0:N][0:M],b[0:M][0:K])
{
    xomp_deviceDataEnvironmentEnter(0);
    float *_dev_a;
    int _dev_a_size[2] = {1024, 1024};
    int _dev_a_offset[2] = {0, 0};
    int _dev_a_Dim[2] = {1024, 1024};
    _dev_a = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)a,2,sizeof(float ),_dev_a_size,_dev_a_offset,_dev_a_Dim,1,0)));
    float *_dev_b;
    int _dev_b_size[2] = {1024, 1024};
    int _dev_b_offset[2] = {0, 0};
    int _dev_b_Dim[2] = {1024, 1024};
    _dev_b = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)b,2,sizeof(float ),_dev_b_size,_dev_b_offset,_dev_b_Dim,1,0)));
    float *_dev_c;
    int _dev_c_size[2] = {1024, 1024};
    int _dev_c_offset[2] = {0, 0};
    int _dev_c_Dim[2] = {1024, 1024};
    _dev_c = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)c,2,sizeof(float ),_dev_c_size,_dev_c_offset,_dev_c_Dim,1,1)));
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
    int _num_blocks_ = xomp_get_max1DBlock(0,1023 - 0 + 1);
    OUT__1__11560__<<<_num_blocks_,_threads_per_block_>>>(_dev_a,_dev_b,_dev_c);
    xomp_deviceDataEnvironmentExit(0);
  }
  return 0;
}

int mmm2()
{
  for (i = 0; i < 1024; i++) 
    for (j = 0; j < 1024; j++) 
      for (k = 0; k < 1024; k++) 
        c2[i][j] = c2[i][j] + a[i][k] * b[k][j];
  return 0;
}

int verify()
{
  float sum = 0.0;
  float sum2 = 0.0;
  for (i = 0; i < 1024; i++) 
    for (j = 0; j < 1024; j++) {
      sum += c[i][j];
      sum2 += c2[i][j];
    }
  printf("sum of c[i][j] is %f\n",sum);
  printf("sum of c2[i][j] is %f\n",sum2);
  sum == sum2?((void )0) : __assert_fail("sum == sum2","matrixmultiply-ompacc.c",94,__PRETTY_FUNCTION__);
  return 0;
}
