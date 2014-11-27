#include<stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 
int a[11][11];

__global__ void OUT__1__7988__(int __final_total_iters__2__,int __i_interval__3__,int *_dev_a)
{
  int _p_i;
  int _p_j;
  int _p___collapsed_index__5__;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(0,__final_total_iters__2__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,__final_total_iters__2__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p___collapsed_index__5__ = _dev_lower; _p___collapsed_index__5__ <= _dev_upper; _p___collapsed_index__5__ += 1) {
      _p_i = _p___collapsed_index__5__ / __i_interval__3__ * 1 + 1;
      _p_j = _p___collapsed_index__5__ % __i_interval__3__ * 1 + 1;
      int k = 3;
      int l = 3;
      int z = 3;
      _dev_a[_p_i * 11 + _p_j] = _p_i + _p_j + l + k + z;
    }
}

int main()
{
  int i;
  int j;
  int m = 10;
  int n = 10;
  int __i_total_iters__0__ = (m - 1 - 1 + 1) % 1 == 0?(m - 1 - 1 + 1) / 1 : (m - 1 - 1 + 1) / 1 + 1;
  int __j_total_iters__1__ = (n - 1 - 1 + 1) % 1 == 0?(n - 1 - 1 + 1) / 1 : (n - 1 - 1 + 1) / 1 + 1;
  int __final_total_iters__2__ = 1 * __i_total_iters__0__ * __j_total_iters__1__;
  int __i_interval__3__ = __j_total_iters__1__ * 1;
  int __j_interval__4__ = 1;
  int __collapsed_index__5__;
  for (i = 0; i < 11; i++) {
    for (j = 0; j < 11; j++) {
      a[i][j] = 0;
    }
  }
{
    int *_dev_a;
    int _dev_a_size = sizeof(int ) * (11 - 0) * (11 - 0);
    _dev_a = ((int *)(xomp_deviceMalloc(_dev_a_size)));
    xomp_memcpyHostToDevice(((void *)_dev_a),((const void *)a),_dev_a_size);
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
    int _num_blocks_ = xomp_get_max1DBlock(__final_total_iters__2__ - 1 - 0 + 1);
    OUT__1__7988__<<<_num_blocks_,_threads_per_block_>>>(__final_total_iters__2__,__i_interval__3__,_dev_a);
    xomp_memcpyDeviceToHost(((void *)a),((const void *)_dev_a),_dev_a_size);
    xomp_freeDevice(_dev_a);
  }
/*
  for(i = 0; i < 11; i ++)
  {
     for(j = 0; j < 11; j ++)
     {
        fprintf(stderr, "%d\n", a[i][j]);
     }
  }
 */
  return 0;
}
