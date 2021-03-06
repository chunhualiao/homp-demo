//[liao6@tux385:~/workspace/autoPar/buildtree/tests/nonsmoke/functional/roseTests/ompLoweringTests]cat rose_jacobi-ompacc-opt2.cu
// Liao, 7/9/2014, add collapse() inside jacobi()
// Liao, 1/22/2015, test nested map() clauses supported by device data environment reuse.
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
  gettimeofday(&t,(struct timezone *)((void *)0));
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
#define MSIZE 5120
int n;
int m;
int mits;
#define REAL float // flexible between float and double
// depending on MSIZE!!
float error_ref = 9.212767E-04;
float resid_ref = 2.355429E-08;
float tol;
float relax = 1.0;
float alpha = 0.0543;
float u[512][512];
float f[512][512];
float uold[512][512];
float dx;
float dy;
// value, reference value, and the number of significant digits to be ensured.

double diff_ratio(double val,double ref,int significant_digits)
{
  significant_digits >= 1?((void )0) : __assert_fail("significant_digits>=1","jacobi-ompacc-opt2.c",67,__PRETTY_FUNCTION__);
  double diff_ratio = fabs(val - ref) / fabs(ref);
// 1.0/(double(10^significant_digits)) ;
  double upper_limit = pow(0.1,significant_digits);
  printf("value :%E  ref_value: %E  diff_ratio: %E upper_limit: %E \n",val,ref,diff_ratio,upper_limit);
// ensure the number of the significant digits to be the same 
  diff_ratio < upper_limit?((void )0) : __assert_fail("diff_ratio < upper_limit","jacobi-ompacc-opt2.c",72,__PRETTY_FUNCTION__);
  return diff_ratio;
}

int main()
{
  xomp_acc_init();
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

__global__ void OUT__1__11053__(float omega,float ax,float ay,float b,int __final_total_iters__2__,int __i_interval__3__,float *_dev_per_block_error,float *_dev_u,float *_dev_f,float *_dev_uold)
{
  int _p_i;
  int _p_j;
  float _p_error;
  _p_error = 0;
  float _p_resid;
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
      _p_resid = (ax * (_dev_uold[(_p_i - 1) * 512 + _p_j] + _dev_uold[(_p_i + 1) * 512 + _p_j]) + ay * (_dev_uold[_p_i * 512 + (_p_j - 1)] + _dev_uold[_p_i * 512 + (_p_j + 1)]) + b * _dev_uold[_p_i * 512 + _p_j] - _dev_f[_p_i * 512 + _p_j]) / b;
      _dev_u[_p_i * 512 + _p_j] = _dev_uold[_p_i * 512 + _p_j] - omega * _p_resid;
      _p_error = _p_error + _p_resid * _p_resid;
    }
  xomp_inner_block_reduction_float(_p_error,_dev_per_block_error,6);
}
// swap old and new arrays
__global__ void OUT__2__11053__(int __final_total_iters__8__,int __i_interval__9__,float *_dev_u,float *_dev_uold)
{
  int _p___collapsed_index__11__;
  int _p_i;
  int _p_j;
  int _dev_lower;
  int _dev_upper;
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(0,__final_total_iters__8__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,__final_total_iters__8__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p___collapsed_index__11__ = _dev_lower; _p___collapsed_index__11__ <= _dev_upper; _p___collapsed_index__11__ += 1) {
      _p_i = _p___collapsed_index__11__ / __i_interval__9__ * 1 + 0;
      _p_j = _p___collapsed_index__11__ % __i_interval__9__ * 1 + 0;
      _dev_uold[_p_i * 512 + _p_j] = _dev_u[_p_i * 512 + _p_j];
    }
}

void jacobi()
{
    float time, cumulative_time = 0.f;
  float omega;
  int k;
  float error;
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
  k = 0;
  // An optimization on top of naive coding: promoting data handling outside the while loop
  // data properties may change since the scope is bigger:
  /* Translated from #pragma omp target data ... */
  {
    xomp_deviceDataEnvironmentEnter(0);
//    float *_dev_u;
    int _dev_u_size[2] = {n, m};
    int _dev_u_offset[2] = {0, 0};
    int _dev_u_Dim[2] = {512, 512};
   // _dev_u = 
       (float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)u,2,sizeof(float ),_dev_u_size,_dev_u_offset,_dev_u_Dim,1,1));
    //float *_dev_f;
    int _dev_f_size[2] = {n, m};
    int _dev_f_offset[2] = {0, 0};
    int _dev_f_Dim[2] = {512, 512};
   // _dev_f = 
    ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)f,2,sizeof(float ),_dev_f_size,_dev_f_offset,_dev_f_Dim,1,0)));
    //float *_dev_uold;
    int _dev_uold_size[2] = {n, m};
    int _dev_uold_offset[2] = {0, 0};
    int _dev_uold_Dim[2] = {512, 512};
    //_dev_uold = 
    ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)uold,2,sizeof(float ),_dev_uold_size,_dev_uold_offset,_dev_uold_Dim,0,0)));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //while(k <= mits && error > tol){
    while(k < 10) {
      int __i_total_iters__0__ = (n - 1 - 1 - 1 + 1) % 1 == 0?(n - 1 - 1 - 1 + 1) / 1 : (n - 1 - 1 - 1 + 1) / 1 + 1;
      int __j_total_iters__1__ = (m - 1 - 1 - 1 + 1) % 1 == 0?(m - 1 - 1 - 1 + 1) / 1 : (m - 1 - 1 - 1 + 1) / 1 + 1;
      int __final_total_iters__2__ = 1 * __i_total_iters__0__ * __j_total_iters__1__;
      int __i_interval__3__ = __j_total_iters__1__ * 1;
      //int __j_interval__4__ = 1;
      //int __collapsed_index__5__;
      int __i_total_iters__6__ = (n - 1 - 0 + 1) % 1 == 0?(n - 1 - 0 + 1) / 1 : (n - 1 - 0 + 1) / 1 + 1;
      int __j_total_iters__7__ = (m - 1 - 0 + 1) % 1 == 0?(m - 1 - 0 + 1) / 1 : (m - 1 - 0 + 1) / 1 + 1;
      int __final_total_iters__8__ = 1 * __i_total_iters__6__ * __j_total_iters__7__;
      int __i_interval__9__ = __j_total_iters__7__ * 1;
      //int __j_interval__10__ = 1;
      //int __collapsed_index__11__;
      error = 0.0;
      /* Copy new solution into old */
      //#pragma omp parallel
      //    {
      {
	xomp_deviceDataEnvironmentEnter(0);
	float *_dev_u;
	int _dev_u_size[2] = {n, m};
	int _dev_u_offset[2] = {0, 0};
	int _dev_u_Dim[2] = {512, 512};
	_dev_u = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)u,2,sizeof(float ),_dev_u_size,_dev_u_offset,_dev_u_Dim,1,0)));
	float *_dev_uold;
	int _dev_uold_size[2] = {n, m};
	int _dev_uold_offset[2] = {0, 0};
	int _dev_uold_Dim[2] = {512, 512};
	_dev_uold = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)uold,2,sizeof(float ),_dev_uold_size,_dev_uold_offset,_dev_uold_Dim,0,1)));
	/* Launch CUDA kernel ... */
	int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
	int _num_blocks_ = xomp_get_max1DBlock(0,__final_total_iters__8__ - 1 - 0 + 1);
	OUT__2__11053__<<<_num_blocks_,_threads_per_block_>>>(__final_total_iters__8__,__i_interval__9__,_dev_u,_dev_uold);
	xomp_deviceDataEnvironmentExit(0);
      }

      // real jacobi kernel calculation

      {
	xomp_deviceDataEnvironmentEnter(0);
	float *_dev_u;
	int _dev_u_size[2] = {n, m};
	int _dev_u_offset[2] = {0, 0};
	int _dev_u_Dim[2] = {512, 512};
	_dev_u = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)u,2,sizeof(float ),_dev_u_size,_dev_u_offset,_dev_u_Dim,0,1)));
	float *_dev_f;
	int _dev_f_size[2] = {n, m};
	int _dev_f_offset[2] = {0, 0};
	int _dev_f_Dim[2] = {512, 512};
	_dev_f = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)f,2,sizeof(float ),_dev_f_size,_dev_f_offset,_dev_f_Dim,1,0)));
	float *_dev_uold;
	int _dev_uold_size[2] = {n, m};
	int _dev_uold_offset[2] = {0, 0};
	int _dev_uold_Dim[2] = {512, 512};
	_dev_uold = ((float *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)uold,2,sizeof(float ),_dev_uold_size,_dev_uold_offset,_dev_uold_Dim,1,0)));
	/* Launch CUDA kernel ... */
	int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
	int _num_blocks_ = xomp_get_max1DBlock(0,__final_total_iters__2__ - 1 - 0 + 1);
	float *_dev_per_block_error = (float *)(xomp_deviceMalloc(_num_blocks_ * sizeof(float )));

cudaEventRecord(start, 0);
if (k==0)
       printf("Kernel launch configuration: 1-D blocks=%d threads-per-block=%d \n", _num_blocks_, _threads_per_block_);
	OUT__1__11053__<<<_num_blocks_,_threads_per_block_,(_threads_per_block_ * sizeof(float ))>>>(omega,ax,ay,b,__final_total_iters__2__,__i_interval__3__,_dev_per_block_error,_dev_u,_dev_f,_dev_uold);
	error = xomp_beyond_block_reduction_float(_dev_per_block_error,_num_blocks_,6);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cumulative_time = cumulative_time + time;


	xomp_freeDevice(_dev_per_block_error);
	xomp_deviceDataEnvironmentExit(0);
      }
      //    }
      /*  omp end parallel */
      k = k + 1;
      /* Error check */
      if (k % 500 == 0) 
	printf("Finished %d iteration with error =%f\n",k,error);
      error = (sqrt(error) / (n * m));
      /*  End iteration loop */
    }
    xomp_deviceDataEnvironmentExit(0);
  }

  printf("array size :%d\n", MSIZE);
   printf("jacobi kernel + reduction time , average over %d times:  %3.5f ms \n", k, cumulative_time / k);
  printf("Total Number of Iterations:%d\n",k);
  printf("Residual:%E\n",error);
  printf("Residual_ref :%E\n",resid_ref);
  printf("Diff ref=%E\n",(fabs((error - resid_ref))));
//  fabs((error - resid_ref)) < 1E-13?((void )0) : __assert_fail("fabs(error-resid_ref) < 1E-13","jacobi-ompacc-opt2.c",247,__PRETTY_FUNCTION__);
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
// turn off this for now , we only iterate 10 times for performance modeling data collection

//  fabs((error - error_ref)) < 1E-13?((void )0) : __assert_fail("fabs(error-error_ref) < 1E-13","jacobi-ompacc-opt2.c",278,__PRETTY_FUNCTION__);
}

