#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

__global__ void OUT__1__4550__(long long nEle,long long m,long long n,int gapScore,int matchScore,int missmatchScore,long long si,long long sj,char *_dev_a,char *_dev_b,int *_dev_H,int *_dev_P,long long *_dev_maxPos_ptr)
{
  long long _p_j;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init((long long )0,nEle - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,nEle - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p_j = _dev_lower; _p_j <= _dev_upper; _p_j += 1) 
// going upwards : anti-diagnol direction
{
// going up vertically
      long long ai = si - _p_j;
//  going right in horizontal
      long long aj = sj + _p_j;
///------------inlined ------------------------------------------
//            similarityScore(ai, aj, H, P, &maxPos); // a critical section is used inside
{
        int up;
        int left;
        int diag;
//Stores index of element
        long long index = m * ai + aj;
//Get element above
        up = _dev_H[index - m - 0] + gapScore;
//Get element on the left
        left = _dev_H[index - ((long long )1) - 0] + gapScore;
//Get element on the diagonal
        int t_mms;
        if (((int )_dev_a[aj - ((long long )1) - 0]) == ((int )_dev_b[ai - ((long long )1) - 0])) 
          t_mms = matchScore;
         else 
          t_mms = missmatchScore;
// matchMissmatchScore(i, j);
        diag = _dev_H[index - m - ((long long )1) - 0] + t_mms;
// degug here
// return;
//Calculates the maximum
        int max = 0;
        int pred = 0;
//same letter ↖
        if (diag > max) {
          max = diag;
          pred = 3;
        }
//remove letter ↑
        if (up > max) {
          max = up;
          pred = 1;
        }
//insert letter ←
        if (left > max) {
          max = left;
          pred = 2;
        }
//Inserts the value in the similarity and predecessor matrixes
        _dev_H[index - 0] = max;
        _dev_P[index - 0] = pred;
//Updates maximum score to be used as seed on backtrack
        if (max > _dev_H[_dev_maxPos_ptr[0] - 0]) {
//#pragma omp critical
          _dev_maxPos_ptr[0 - 0] = index;
        }
      }
// ---------------------------------------------------------------
    }
}

void calculate(char *a,char *b,long long nEle,long long m,long long n,int gapScore,int matchScore,int missmatchScore,long long si,long long sj,int *H,int *P,long long *maxPos_ptr,long long j,int asz)
{
{
    xomp_deviceDataEnvironmentEnter(0);
    char *_dev_a;
    int _dev_a_size[1] = {m};
    int _dev_a_offset[1] = {0};
    int _dev_a_Dim[1] = {m};
    _dev_a = ((char *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)a,1,sizeof(char ),_dev_a_size,_dev_a_offset,_dev_a_Dim,1,0)));
    char *_dev_b;
    int _dev_b_size[1] = {n};
    int _dev_b_offset[1] = {0};
    int _dev_b_Dim[1] = {n};
    _dev_b = ((char *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)b,1,sizeof(char ),_dev_b_size,_dev_b_offset,_dev_b_Dim,1,0)));
    int *_dev_H;
    int _dev_H_size[1] = {asz};
    int _dev_H_offset[1] = {0};
    int _dev_H_Dim[1] = {asz};
    _dev_H = ((int *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)H,1,sizeof(int ),_dev_H_size,_dev_H_offset,_dev_H_Dim,1,1)));
    int *_dev_P;
    int _dev_P_size[1] = {asz};
    int _dev_P_offset[1] = {0};
    int _dev_P_Dim[1] = {asz};
    _dev_P = ((int *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)P,1,sizeof(int ),_dev_P_size,_dev_P_offset,_dev_P_Dim,1,1)));
    long long *_dev_maxPos_ptr;
    int _dev_maxPos_ptr_size[1] = {1};
    int _dev_maxPos_ptr_offset[1] = {0};
    int _dev_maxPos_ptr_Dim[1] = {1};
    _dev_maxPos_ptr = ((long long *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)maxPos_ptr,1,sizeof(long long ),_dev_maxPos_ptr_size,_dev_maxPos_ptr_offset,_dev_maxPos_ptr_Dim,1,1)));
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
    int _num_blocks_ = xomp_get_max1DBlock(0,nEle - 1 - ((long long )0) + 1);
    OUT__1__4550__<<<_num_blocks_,_threads_per_block_>>>(nEle,m,n,gapScore,matchScore,missmatchScore,si,sj,_dev_a,_dev_b,_dev_H,_dev_P,_dev_maxPos_ptr);
    xomp_deviceDataEnvironmentExit(0);
  }
}
//      } // for end nDiag
//    } // end omp parallel
//}
