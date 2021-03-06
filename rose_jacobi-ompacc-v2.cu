// Naive version without any optimizations
// test mapped static arrays without array section info specified.  
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// Add timing support
#include <sys/time.h>
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

double time_stamp()
{
  struct timeval t;
  double time;
  gettimeofday(&t,((struct timezone *)((void *)0)));
  time = t . tv_sec + 1.0e-6 * t . tv_usec;
  return time;
}
double time1;
double time2;
void driver();
void initialize();
void jacobi();
void error_check();
/************************************************************
* program to solve a finite difference 
* discretization of Helmholtz equation :  
* (d2/dx2)u + (d2/dy2)u - alpha u = f 
* using Jacobi iterative method. 
*
* Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
* Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
*
* This c version program is translated by 
* Chunhua Liao, University of Houston, Jan, 2005 
* 
* Directives are used in this code to achieve parallelism. 
* All do loops are parallelized with default 'static' scheduling.
* 
* Input :  n - grid dimension in x direction 
*          m - grid dimension in y direction
*          alpha - Helmholtz constant (always greater than 0.0)
*          tol   - error tolerance for iterative solver
*          relax - Successice over relaxation parameter
*          mits  - Maximum iterations for iterative solver
*
* On output 
*       : u(n,m) - Dependent variable (solutions)
*       : f(n,m) - Right hand side function 
*************************************************************/
#define REAL float // flexible between float and double
#define MSIZE 512
// depending on MSIZE!!
float error_ref = 9.212767E-04;
float resid_ref = 2.355429E-08;
int n;
int m;
int mits;
float tol;
float relax = 1.0;
float alpha = 0.0543;
float u[512][512];
float f[512][512];
float uold[512][512];
float dx;
float dy;

int main()
{
//  float toler;
/*      printf("Input n,m (< %d) - grid dimension in x,y direction:\n",MSIZE); 
          scanf ("%d",&n);
          scanf ("%d",&m);
          printf("Input tol - error tolerance for iterative solver\n"); 
          scanf("%f",&toler);
          tol=(double)toler;
          printf("Input mits - Maximum iterations for solver\n"); 
          scanf("%d",&mits);
          */
  n = 512;
  m = 512;
  tol = 0.0000000001;
  mits = 5000;
#if 0 // Not yet support concurrent CPU and GPU threads  
#ifdef _OPENMP
#endif
#endif  
  driver();
  return 0;
}
/*************************************************************
* Subroutine driver () 
* This is where the arrays are allocated and initialzed. 
*
* Working varaibles/arrays 
*     dx  - grid spacing in x direction 
*     dy  - grid spacing in y direction 
*************************************************************/

void driver()
{
  initialize();
  time1 = time_stamp();
/* Solve Helmholtz equation */
  jacobi();
  time2 = time_stamp();
  printf("------------------------\n");
  printf("Execution time = %f\n",time2 - time1);
/* error_check (n,m,alpha,dx,dy,u,f)*/
  error_check();
}
/*      subroutine initialize (n,m,alpha,dx,dy,u,f) 
******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/

void initialize()
{
  int i;
  int j;
  int xx;
  int yy;
//double PI=3.1415926;
  dx = (2.0 / (n - 1));
  dy = (2.0 / (m - 1));
/* Initialize initial condition and RHS */
//#pragma omp parallel for private(xx,yy,j,i)
  for (i = 0; i < n; i++) 
    for (j = 0; j < m; j++) {
      xx = ((int )(- 1.0 + (dx * (i - 1))));
      yy = ((int )(- 1.0 + (dy * (j - 1))));
      u[i][j] = 0.0;
      f[i][j] = (- 1.0 * alpha * (1.0 - (xx * xx)) * (1.0 - (yy * yy)) - 2.0 * (1.0 - (xx * xx)) - 2.0 * (1.0 - (yy * yy)));
    }
}
/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,maxit)
******************************************************************
* Subroutine HelmholtzJ
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method is used in this routine 
*
* Input : n,m   Number of grid points in the X/Y directions 
*         dx,dy Grid spacing in the X/Y directions 
*         alpha Helmholtz eqn. coefficient 
*         omega Relaxation factor 
*         f(n,m) Right hand side function 
*         u(n,m) Dependent variable/Solution
*         tol    Tolerance for iterative solver 
*         maxit  Maximum number of iterations 
*
* Output : u(n,m) - Solution 
*****************************************************************/

__global__ void OUT__1__8493__(int n,int m,float omega,float ax,float ay,float b,float *_dev_per_block_error,float *_dev_u,float *_dev_f,float *_dev_uold)
{
  int _p_i;
  int _p_j;
  float _p_error;
  _p_error = 0;
  float _p_resid;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(1,n - 1 - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,n - 1 - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
      for (_p_j = 1; _p_j < m - 1; _p_j++) {
        _p_resid = (ax * (_dev_uold[(_p_i - 1) * 512 + _p_j] + _dev_uold[(_p_i + 1) * 512 + _p_j]) + ay * (_dev_uold[_p_i * 512 + (_p_j - 1)] + _dev_uold[_p_i * 512 + (_p_j + 1)]) + b * _dev_uold[_p_i * 512 + _p_j] - _dev_f[_p_i * 512 + _p_j]) / b;
        _dev_u[_p_i * 512 + _p_j] = _dev_uold[_p_i * 512 + _p_j] - omega * _p_resid;
        _p_error = _p_error + _p_resid * _p_resid;
      }
    }
  xomp_inner_block_reduction_float(_p_error,_dev_per_block_error,6);
}

__global__ void OUT__2__8493__(int n,int m,float *_dev_u,float *_dev_uold)
{
  int _p_i;
  int _p_j;
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
      for (_p_j = 0; _p_j < m; _p_j++) 
        _dev_uold[_p_i * 512 + _p_j] = _dev_u[_p_i * 512 + _p_j];
    }
}

void jacobi()
{
  float omega;
  int i;
  int j;
  int k;
  float error;
  float resid;
  float ax;
  float ay;
  float b;
//      double  error_local;
//      float ta,tb,tc,td,te,ta1,ta2,tb1,tb2,tc1,tc2,td1,td2;
//      float te1,te2;
//      float second;
  omega = relax;
/*
   * Initialize coefficients */
/* X-direction coef */
  ax = (1.0 / (dx * dx));
/* Y-direction coef */
  ay = (1.0 / (dy * dy));
/* Central coeff */
  b = (- 2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha);
  error = (10.0 * tol);
  k = 1;
  while(k <= mits && error > tol){
    error = 0.0;
/* Copy new solution into old */
{
      xomp_deviceDataEnvironmentEnter();
      float *_dev_u;
      int _dev_u_size = sizeof(float ) * 512 * 512;
      _dev_u = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(((void *)u),_dev_u_size,1,0)));
      float *_dev_uold;
      int _dev_uold_size = sizeof(float ) * 512 * 512;
      _dev_uold = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(((void *)uold),_dev_uold_size,0,1)));
/* Launch CUDA kernel ... */
      int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
      int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
      OUT__2__8493__<<<_num_blocks_,_threads_per_block_>>>(n,m,_dev_u,_dev_uold);
      xomp_deviceDataEnvironmentExit();
    }
{
      xomp_deviceDataEnvironmentEnter();
      float *_dev_u;
      int _dev_u_size = sizeof(float ) * 512 * 512;
      _dev_u = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(((void *)u),_dev_u_size,0,1)));
      float *_dev_f;
      int _dev_f_size = sizeof(float ) * 512 * 512;
      _dev_f = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(((void *)f),_dev_f_size,1,0)));
      float *_dev_uold;
      int _dev_uold_size = sizeof(float ) * 512 * 512;
      _dev_uold = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(((void *)uold),_dev_uold_size,1,0)));
/* Launch CUDA kernel ... */
      int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
      int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 1 - 1 + 1);
      float *_dev_per_block_error = (float *)(xomp_deviceMalloc(_num_blocks_ * sizeof(float )));
      OUT__1__8493__<<<_num_blocks_,_threads_per_block_,(_threads_per_block_ * sizeof(float ))>>>(n,m,omega,ax,ay,b,_dev_per_block_error,_dev_u,_dev_f,_dev_uold);
      error = xomp_beyond_block_reduction_float(_dev_per_block_error,_num_blocks_,6);
      xomp_freeDevice(_dev_per_block_error);
      xomp_deviceDataEnvironmentExit();
    }
//    }
/*  omp end parallel */
/* Error check */
    if (k % 500 == 0) {
      printf("Finished %d iteration with error =%f\n",k,error);
    }
    error = (sqrt(error) / (n * m));
    k = k + 1;
/*  End iteration loop */
  }
  printf("Total Number of Iterations:%d\n",k);
  printf("Residual:%E\n",error);
  printf("Residual_ref :%E\n",resid_ref);
  printf("Diff ref=%E\n",(fabs((error - resid_ref))));
  fabs((error - resid_ref)) < 1E-14?((void )0) : __assert_fail("fabs(error-resid_ref) < 1E-14","jacobi-ompacc-v2.c",236,__PRETTY_FUNCTION__);
}
/*      subroutine error_check (n,m,alpha,dx,dy,u,f) 
      implicit none 
************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/

void error_check()
{
  int i;
  int j;
  float xx;
  float yy;
  float temp;
  float error;
  dx = (2.0 / (n - 1));
  dy = (2.0 / (m - 1));
  error = 0.0;
//#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
  for (i = 0; i < n; i++) 
    for (j = 0; j < m; j++) {
      xx = (- 1.0 + (dx * (i - 1)));
      yy = (- 1.0 + (dy * (j - 1)));
      temp = (u[i][j] - (1.0 - (xx * xx)) * (1.0 - (yy * yy)));
      error = error + temp * temp;
    }
  error = (sqrt(error) / (n * m));
  printf("Solution Error :%E \n",error);
  printf("Solution Error Ref :%E \n",error_ref);
  printf("Diff ref=%E\n",(fabs((error - error_ref))));
  fabs((error - error_ref)) < 1E-14?((void )0) : __assert_fail("fabs(error-error_ref) < 1E-14","jacobi-ompacc-v2.c",268,__PRETTY_FUNCTION__);
}
