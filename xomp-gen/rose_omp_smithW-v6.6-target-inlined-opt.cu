//static void *xomp_critical_user_;
/*********************************************************************************
 * Smith–Waterman algorithm
 * Purpose:     Local alignment of nucleotide or protein sequences
 * Authors:     Daniel Holanda, Hanoch Griner, Taynara Pinheiro
 *
 * Compilation: gcc omp_smithW.c -o omp_smithW -fopenmp -DDEBUG // debugging mode
 *              gcc omp_smithW.c -O3 -o omp_smithW -fopenmp // production run
 * Execution:	./omp_smithW <number_of_col> <number_of_rows>
 *
 * Updated by C. Liao, Jan 2nd, 2019
 *********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <omp.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h> // C99 does not support the boolean data type
//#include "parameters.h"
#define FACTOR 128 
#define CUTOFF 1024
/*--------------------------------------------------------------------
 * Text Tweaks
 */
#define RESET   "\033[0m"
#define BOLDRED "\033[1m\033[31m"      /* Bold Red */
/* End of text tweaks */
/*--------------------------------------------------------------------
 * Constants
 */
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
/* End of constants */
/*--------------------------------------------------------------------
* Helpers
*/
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(a,b) ((a) > (b) ? a : b)
// #define DEBUG
/* End of Helpers */
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
#if 1
// extern double omp_get_wtime (void) __GOMP_NOTHROW;
double omp_get_wtime() 
{
  return time_stamp();
}
#endif
/*--------------------------------------------------------------------
 * Functions Prototypes
 */
//Defines size of strings to be compared
//Columns - Size of string a
long long m = 8;
//Lines - Size of string b
long long n = 9;
int gapScore = - 2;
//Defines scores
int matchScore = 3;
int missmatchScore = - 3;
//Strings over the Alphabet Sigma
char *a;
char *b;
int matchMissmatchScore(long long i,long long j);
void similarityScore(long long i,long long j,int *H,int *P,long long *maxPos);
// without omp critical: how to conditionalize it?
void similarityScore2(long long i,long long j,int *H,int *P,long long *maxPos);
void backtrack(int *P,long long maxPos);
void printMatrix(int *matrix);
void printPredecessorMatrix(int *matrix);
void generate();
long long nElement(long long i);
void calcFirstDiagElement(long long i,long long *si,long long *sj);
/* End of prototypes */
/*--------------------------------------------------------------------
 * Global Variables
 */
_Bool useBuiltInData = 1;
// the generated scoring matrix's size is m++ and n++ later to have the first row/column as 0s.
/* End of global variables */
/*--------------------------------------------------------------------
 * Function:    main
 */

__global__ void OUT__1__7018__(long long m,long long n,int gapScore,int matchScore,int missmatchScore,long long nEle,long long si,long long sj,char *_dev_a,char *_dev_b,int *_dev_H,int *_dev_P,long long *_dev_maxPos_ptr)
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
#if !SKIP_BACKTRACK
        //Updates maximum score to be used as seed on backtrack
        {   
          // \note \pp
          //   locks seem to be a NOGO in CUDA warps,
          //   thus the update to set the maximum is made nonblocking.
          unsigned long long int current = _dev_maxPos_ptr[0];
          unsigned long long int assumed = current+1;

          while (assumed != current && max > _dev_H[current])
          { 
            assumed = current;

            // \note consider atomicCAS_system for multi GPU systems
            current = atomicCAS((unsigned long long int*)_dev_maxPos_ptr, (unsigned long long int)assumed, (unsigned long long int)index);
          } 
        }   
#endif        
      } // end inlined similarityScore()
      // ---------------------------------------------------------------
    } // end for loop
}

int main(int argc,char *argv[])
{
  xomp_acc_init();
  // thread_count is no longer used
  //  int thread_count;
  if (argc == 3) {
    m = strtoll(argv[1],(char**)((void *)0),10);
    n = strtoll(argv[2],(char**)((void *)0),10);
    useBuiltInData = 0;
  }
  //#ifdef DEBUG
  if (useBuiltInData) 
    printf("Using built-in data for testing ..\n");
  printf("Problem size: Matrix[%lld][%lld], FACTOR=%d CUTOFF=%d\n",n,m,128,1024);
  //#endif
  //Allocates a and b
  a = ((char *)(malloc((m * (sizeof(char ))))));
  //    printf ("debug: a's address=%p\n", a);
  b = ((char *)(malloc((n * (sizeof(char ))))));
  //    printf ("debug: b's address=%p\n", b);
  //Because now we have zeros
  m++;
  n++;
  //Allocates similarity matrix H
  int *H;
  H = ((int *)(calloc((m * n),sizeof(int ))));
  //    printf ("debug: H's address=%p\n", H);
  //Allocates predecessor matrix P
  int *P;
  P = ((int *)(calloc((m * n),sizeof(int ))));
  //    printf ("debug: P's address=%p\n", P);
  if (useBuiltInData) {
    // https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Example
    // Using the wiki example to verify the results
    b[0] = 'G';
    b[1] = 'G';
    b[2] = 'T';
    b[3] = 'T';
    b[4] = 'G';
    b[5] = 'A';
    b[6] = 'C';
    b[7] = 'T';
    b[8] = 'A';
    a[0] = 'T';
    a[1] = 'G';
    a[2] = 'T';
    a[3] = 'T';
    a[4] = 'A';
    a[5] = 'C';
    a[6] = 'G';
    a[7] = 'G';
  }
  else {
    //Gen random arrays a and b
    generate();
  }
  //Start position for backtrack
  long long maxPos = 0;
  //Calculates the similarity matrix
  long long i;
  //  long long j;
  // The way to generate all wavefront is to go through the top edge elements
  // starting from the left top of the matrix, go to the bottom top -> down, then left->right
  // total top edge element count =  dim1_size + dim2_size -1 
  //Because now we have zeros ((m-1) + (n-1) - 1)
  long long nDiag = m + n - 3;
  //Gets Initial time
  double initialTime = omp_get_wtime();
  // mistake: element count, not byte size!!
  // int asz= m*n*sizeof(int);
  int asz = (m * n);
  // choice 2: map data before the outer loop
  //#pragma omp target map (to:a[0:m], b[0:n], nDiag, m,n,gapScore, matchScore, missmatchScore) map(tofrom: H[0:asz], P[0:asz], maxPos)
  //--------------------------------------
  // data mapping begins
  // choice 1: map data before the inner loop
  long long *maxPos_ptr = &maxPos;
  xomp_deviceDataEnvironmentEnter(0);
  char *_dev_a;
  int _dev_a_size[1] = {(int)m};
  int _dev_a_offset[1] = {0};
  int _dev_a_Dim[1] = {(int)m};
  _dev_a = ((char *)(xomp_deviceDataEnvironmentPrepareVariable(0,(void *)a,1,sizeof(char ),_dev_a_size,_dev_a_offset,_dev_a_Dim,1,0)));
  char *_dev_b;
  int _dev_b_size[1] = {(int)n};
  int _dev_b_offset[1] = {0};
  int _dev_b_Dim[1] = {(int)n};
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

  // data mapping ends

  //  #pragma omp parallel default(none) shared(H, P, maxPos, nDiag, j) private(i)
  {
    // start from 1 since 0 is the boundary padding
    for (i = 1; i <= nDiag; ++i) {
      long long nEle;
      long long si;
      long long sj;
      //  nEle = nElement(i);
      //---------------inlined ------------
      // smaller than both directions
      if (i < m && i < n) {
        //Number of elements in the diagonal is increasing
        nEle = i;
      }
      else 
        // smaller than only one direction
        if (i < ((m > n?m : n))) {
          //Number of elements in the diagonal is stable
          // the longer direction has the edge elements, the number is the smaller direction's size
          long min = (m < n?m : n);
          nEle = (min - 1);
        }
        else {
          //Number of elements in the diagonal is decreasing
          long min = (m < n?m : n);
          nEle = (2 * min) - i + llabs(m - n) - 2;
        }
      //calcFirstDiagElement(i, &si, &sj);
      //------------inlined---------------------
      // Calculate the first element of diagonal
      // smaller than row count
      if (i < n) {
        si = i;
        // start from the j==1 since j==0 is the padding
        sj = 1;
        // now we sweep horizontally at the bottom of the matrix
      }
      else {
        // i is fixed
        si = n - 1;
        // j position is the nDiag (id -n) +1 +1 // first +1 
        sj = i - n + 2;
      }
      // for end nDiag
      /* Launch CUDA kernel ... */
      int _threads_per_block_ = xomp_get_maxThreadsPerBlock(0);
      int _num_blocks_ = xomp_get_max1DBlock(0,nEle - 1 - ((long long )0) + 1);
      OUT__1__7018__<<<_num_blocks_,_threads_per_block_>>>(m,n,gapScore,matchScore,missmatchScore,nEle,si,sj,_dev_a,_dev_b,_dev_H,_dev_P,_dev_maxPos_ptr);

    }
    // end omp parallel
  }
  xomp_deviceDataEnvironmentExit(0);

  double finalTime = omp_get_wtime();
  printf("\nElapsed time for scoring matrix computation: %f\n",finalTime - initialTime);
#if !SKIP_BACKTRACK  
  initialTime = omp_get_wtime();
  backtrack(P,maxPos);
  finalTime = omp_get_wtime();
  //Gets backtrack time
  finalTime = omp_get_wtime();
  printf("Elapsed time for backtracking: %f\n",finalTime - initialTime);
#endif
  if (useBuiltInData) {
    printf("Verifying results using the builtinIn data: %s\n",(H[n * m - 1] == 7?"true" : "false"));
    assert (H[n * m - 1] == 7);
  }
  //Frees similarity matrixes
  free(H);
  free(P);
#if 0  // causing core dump for larger arrays
  //Frees input arrays
  free(a);
  free(b);
#endif   
  return 0;
  /* End of main */
}
/*--------------------------------------------------------------------
 * Function:    nElement
 * Purpose:     Calculate the number of i-diagonal's elements
 * i value range 1 to nDiag.  we inclulde the upper bound value. 0 is for the padded wavefront, which is ignored.
 */

long long nElement(long long i)
{
// smaller than both directions
  if (i < m && i < n) {
//Number of elements in the diagonal is increasing
    return i;
  }
   else 
// smaller than only one direction
if (i < ((m > n?m : n))) {
//Number of elements in the diagonal is stable
// the longer direction has the edge elements, the number is the smaller direction's size
    long min = (m < n?m : n);
    return (min - 1);
  }
   else {
//Number of elements in the diagonal is decreasing
    long min = (m < n?m : n);
    return (2 * min) - i + llabs(m - n) - 2;
  }
}
/*--------------------------------------------------------------------
 * Function:    calcElement: expect valid i value is from 1 to nDiag. since the first one is 0 padding
 * Purpose:     Calculate the position of (si, sj)-element
 * n rows, m columns: we sweep the matrix on the left edge then bottom edge to get the wavefront
 */

void calcFirstDiagElement(long long i,long long *si,long long *sj)
{
// Calculate the first element of diagonal
// smaller than row count
  if (i < n) {
     *si = i;
// start from the j==1 since j==0 is the padding
     *sj = 1;
// now we sweep horizontally at the bottom of the matrix
  }
   else {
// i is fixed
     *si = n - 1;
// j position is the nDiag (id -n) +1 +1 // first +1 
     *sj = i - n + 2;
  }
}
/*
 // understanding the calculation by an example
 n =6 // row
 m =2  // col
 padded scoring matrix
 n=7
 m=3
   0 1 2
 -------
 0 x x x
 1 x x x
 2 x x x
 3 x x x
 4 x x x
 5 x x x
 6 x x x
 We should peel off top row and left column since they are the padding
 the remaining 6x2 sub matrix is what is interesting for us
 Now find the number of wavefront lines and their first element's position in the scoring matrix
total diagnol frontwave = (n-1) + (m-1) -1 // submatrix row+column -1
We use the left most element in each wavefront line as its first element.
Then we have the first elements like
(1,1),
(2,1)
(3,1)
..
(6,1) (6,2)
 
 */
/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  value of scoring matrix element H(i,j) : the maximum Similarity-Score H(i,j)
 *             int *P; the predecessor array,storing which of the three elements is picked with max value
 */

void similarityScore(long long i,long long j,int *H,int *P,long long *maxPos)
{
  int up;
  int left;
  int diag;
//Stores index of element
  long long index = m * i + j;
//Get element above
  up = H[index - m] + gapScore;
//Get element on the left
  left = H[index - 1] + gapScore;
//Get element on the diagonal
  int t_mms;
  if (a[j - 1] == b[i - 1]) 
    t_mms = matchScore;
   else 
    t_mms = missmatchScore;
// matchMissmatchScore(i, j);
  diag = H[index - m - 1] + t_mms;
// degug here
// return;
//Calculates the maximum
  int max = 0;
  int pred = 0;
/* === Matrix ===
     *      a[0] ... a[n]
     * b[0]
     * ...
     * b[n]
     *
     * generate 'a' from 'b', if '←' insert e '↑' remove
     * a=GAATTCA
     * b=GACTT-A
     *
     * generate 'b' from 'a', if '←' insert e '↑' remove
     * b=GACTT-A
     * a=GAATTCA
    */
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
  H[index] = max;
  P[index] = pred;
//Updates maximum score to be used as seed on backtrack
  if (max > H[ *maxPos]) {
    //XOMP_critical_start(&xomp_critical_user_);
     *maxPos = index;
    //XOMP_critical_end(&xomp_critical_user_);
  }
/* End of similarityScore */
}
/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */

int matchMissmatchScore(long long i,long long j)
{
  if (a[j - 1] == b[i - 1]) 
    return matchScore;
   else 
    return missmatchScore;
/* End of matchMissmatchScore */
}

void similarityScore2(long long i,long long j,int *H,int *P,long long *maxPos)
{
  int up;
  int left;
  int diag;
//Stores index of element
  long long index = m * i + j;
//Get element above
  up = H[index - m] + gapScore;
//Get element on the left
  left = H[index - 1] + gapScore;
//Get element on the diagonal
  diag = H[index - m - 1] + matchMissmatchScore(i,j);
//Calculates the maximum
  int max = 0;
  int pred = 0;
/* === Matrix ===
     *      a[0] ... a[n]
     * b[0]
     * ...
     * b[n]
     *
     * generate 'a' from 'b', if '←' insert e '↑' remove
     * a=GAATTCA
     * b=GACTT-A
     *
     * generate 'b' from 'a', if '←' insert e '↑' remove
     * b=GACTT-A
     * a=GAATTCA
    */
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
  H[index] = max;
  P[index] = pred;
//Updates maximum score to be used as seed on backtrack
  if (max > H[ *maxPos]) {
     *maxPos = index;
  }
/* End of similarityScore2 */
}
/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */

void backtrack(int *P,long long maxPos)
{
//hold maxPos value
  long long predPos;
//backtrack from maxPos to startPos = 0
  do {
    if (P[maxPos] == 3) 
      predPos = maxPos - m - 1;
     else if (P[maxPos] == 1) 
      predPos = maxPos - m;
     else if (P[maxPos] == 2) 
      predPos = maxPos - 1;
    P[maxPos] *= - 1;
    maxPos = predPos;
  }while (P[maxPos] != 0);
/* End of backtrack */
}
/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */

void printMatrix(int *matrix)
{
  long long i;
  long long j;
  printf("-\t-\t");
  for (j = 0; j < m - 1; j++) {
    printf("%c\t",a[j]);
  }
  printf("\n-\t");
//Lines
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (j == 0 && i > 0) 
        printf("%c\t",b[i - 1]);
      printf("%d\t",matrix[m * i + j]);
    }
    printf("\n");
  }
/* End of printMatrix */
}
/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */

void printPredecessorMatrix(int *matrix)
{
  long long i;
  long long j;
  long long index;
  printf("    ");
  for (j = 0; j < m - 1; j++) {
    printf("%c ",a[j]);
  }
  printf("\n  ");
//Lines
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (j == 0 && i > 0) 
        printf("%c ",b[i - 1]);
      index = m * i + j;
      if (matrix[index] < 0) {
        printf("\033[1m\033[31m");
        if (matrix[index] == - 1) 
          printf("\342\206\221 ");
         else if (matrix[index] == - 2) 
          printf("\342\206\220 ");
         else if (matrix[index] == - 3) 
          printf("\342\206\226 ");
         else 
          printf("- ");
        printf("\033[0m");
      }
       else {
        if (matrix[index] == 1) 
          printf("\342\206\221 ");
         else if (matrix[index] == 2) 
          printf("\342\206\220 ");
         else if (matrix[index] == 3) 
          printf("\342\206\226 ");
         else 
          printf("- ");
      }
    }
    printf("\n");
  }
/* End of printPredecessorMatrix */
}
/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */

void generate()
{
//Random seed
  srand((time((time_t*)((void *)0))));
//Generates the values of a
  long long i;
  for (i = 0; i < m; i++) {
    int aux = rand() % 4;
    if (aux == 0) 
      a[i] = 'A';
     else if (aux == 2) 
      a[i] = 'C';
     else if (aux == 3) 
      a[i] = 'G';
     else 
      a[i] = 'T';
  }
//Generates the values of b
  for (i = 0; i < n; i++) {
    int aux = rand() % 4;
    if (aux == 0) 
      b[i] = 'A';
     else if (aux == 2) 
      b[i] = 'C';
     else if (aux == 3) 
      b[i] = 'G';
     else 
      b[i] = 'T';
  }
/* End of generate */
}
/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */
