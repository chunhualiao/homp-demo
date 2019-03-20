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
#include <omp.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h> // C99 does not support the boolean data type
#include "parameters.h"
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
#ifndef _OPENMP
#include <sys/time.h>
#endif
#include "libxomp.h" 
extern void calculate(char *,char *,long long ,long long ,long long ,int ,int ,int ,long long ,long long ,int *,int *,long long *,long long ,int );
/*--------------------------------------------------------------------
 * Functions Prototypes
 */
//#pragma omp declare target
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
//#pragma omp end declare target
// without omp critical: how to conditionalize it?
void similarityScore2(long long i,long long j,int *H,int *P,long long *maxPos);
void backtrack(int *P,long long maxPos);
void printMatrix(int *matrix);
void printPredecessorMatrix(int *matrix);
void generate();
long long nElement(long long i);
void calcFirstDiagElement(long long i,long long *si,long long *sj);
double time_stamp()
{
  struct timeval t;
  double time;
  gettimeofday(&t,(struct timezone *)((void *)0));
  time = t . tv_sec + 1.0e-6 * t . tv_usec;
  return time;
}

double omp_get_wtime()
{
  return time_stamp();
}
/* End of prototypes */
/*--------------------------------------------------------------------
 * Global Variables
 */
_Bool useBuiltInData = 1;
int MEDIUM = 10240;
// max 46340 for GPU of 16GB Device memory
int LARGE = 20480;
// the generated scoring matrix's size is m++ and n++ later to have the first row/column as 0s.
/* End of global variables */
/*--------------------------------------------------------------------
 * Function:    main
 */

struct OUT__1__4435___data 
{
  void *H_p;
  void *P_p;
  void *maxPos_p;
  void *nEle_p;
  void *si_p;
  void *sj_p;
}
;
static void OUT__1__4435__(void *__out_argv);

int main(int argc,char *argv[])
{
  int status = 0;
  XOMP_init(argc,argv);
// thread_count is no longer used
  int thread_count;
  if (argc == 3) {
    m = strtoll(argv[1],((void *)0),10);
    n = strtoll(argv[2],((void *)0),10);
    useBuiltInData = 0;
  }
//#ifdef DEBUG
  if (useBuiltInData) {
    printf("Usage: %s m n\n",argv[0]);
    printf("Using built-in data for testing ..\n");
  }
  printf("Problem size: Matrix[%lld][%lld], Medium=%d Large=%d\n",n,m,MEDIUM,LARGE);
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
//Uncomment this to test the sequence available at 
//http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
// OBS: m=11 n=7
// a[0] =   'C';
// a[1] =   'G';
// a[2] =   'T';
// a[3] =   'G';
// a[4] =   'A';
// a[5] =   'A';
// a[6] =   'T';
// a[7] =   'T';
// a[8] =   'C';
// a[9] =   'A';
// a[10] =  'T';
// b[0] =   'G';
// b[1] =   'A';
// b[2] =   'C';
// b[3] =   'T';
// b[4] =   'T';
// b[5] =   'A';
// b[6] =   'C';
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
  long long j;
// The way to generate all wavefront is to go through the top edge elements
// starting from the left top of the matrix, go to the bottom top -> down, then left->right
// total top edge element count =  dim1_size + dim2_size -1 
//Because now we have zeros ((m-1) + (n-1) - 1)
  long long nDiag = m + n - 3;
#ifdef DEBUG
#endif
/*
#ifdef _OPENMP
#pragma omp parallel 
    {
#pragma omp master	    
      {
        thread_count = omp_get_num_threads();
        printf ("Using %d out of max %d threads...\n", thread_count, omp_get_max_threads());
      }
    }
#endif
*/
//Gets Initial time
  double initialTime = omp_get_wtime();
// mistake: element count, not byte size!!
// int asz= m*n*sizeof(int);
  int asz = (m * n);
// choice 2: map data before the outer loop
//#pragma omp target map (to:a[0:m], b[0:n], nDiag, m,n,gapScore, matchScore, missmatchScore) map(tofrom: H[0:asz], P[0:asz], maxPos)
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
// serial version: 0 to < medium: small data set
      if (nEle < MEDIUM) {
        for (j = 0; j < nEle; ++j) 
// going upwards : anti-diagnol direction
{
// going up vertically
          long long ai = si - j;
//  going right in horizontal
          long long aj = sj + j;
// a specialized version without a critical section used inside
          similarityScore2(ai,aj,H,P,&maxPos);
        }
      }
       else 
// omp cpu version: medium to large: medium data set
if (nEle < LARGE) {
        long long *maxPos_ptr = &maxPos;
        struct OUT__1__4435___data __out_argv1__4435__;
        __out_argv1__4435__ . sj_p = ((void *)(&sj));
        __out_argv1__4435__ . si_p = ((void *)(&si));
        __out_argv1__4435__ . nEle_p = ((void *)(&nEle));
        __out_argv1__4435__ . maxPos_p = ((void *)(&maxPos));
        __out_argv1__4435__ . P_p = ((void *)(&P));
        __out_argv1__4435__ . H_p = ((void *)(&H));
        XOMP_parallel_start(OUT__1__4435__,&__out_argv1__4435__,1,0,"/rose/rose_build/exampleTranslators/test/v7.c",295);
        XOMP_parallel_end("/rose/rose_build/exampleTranslators/test/v7.c",301);
      }
       else 
// omp gpu version: large data set
//--------------------------------------
{
// choice 1: map data before the inner loop
        long long *maxPos_ptr = &maxPos;
        calculate(a,b,nEle,m,n,gapScore,matchScore,missmatchScore,si,sj,H,P,maxPos_ptr,j,asz);
      }
    }
  }
  double finalTime = omp_get_wtime();
  printf("\nElapsed time for scoring matrix computation: %f\n",finalTime - initialTime);
  initialTime = omp_get_wtime();
  backtrack(P,maxPos);
  finalTime = omp_get_wtime();
//Gets backtrack time
  finalTime = omp_get_wtime();
  printf("Elapsed time for backtracking: %f\n",finalTime - initialTime);
#ifdef DEBUG
#endif
  if (useBuiltInData) {
    printf("Verifying results using the builtinIn data: %s\n",(H[n * m - 1] == 7?"true" : "false"));
    H[n * m - 1] == 7?((void )0) : __assert_fail("H[n*m-1]==7","v7.c",335,__PRETTY_FUNCTION__);
  }
//Frees similarity matrixes
  free(H);
  free(P);
//Frees input arrays
  free(a);
  free(b);
  XOMP_terminate(status);
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
// #pragma omp declare target

void similarityScore(long long i,long long j,int *H,int *P,long long *maxPos_ptr)
{
//long long int *maxPos_ptr = &maxPos;
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
  if (max > H[maxPos_ptr[0]]) {
//#pragma omp critical
    maxPos_ptr[0] = index;
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
//#pragma omp end declare target

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
  srand((time(((void *)0))));
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

static void OUT__1__4435__(void *__out_argv)
{
  int **H = (int **)(((struct OUT__1__4435___data *)__out_argv) -> H_p);
  int **P = (int **)(((struct OUT__1__4435___data *)__out_argv) -> P_p);
  long long *maxPos = (long long *)(((struct OUT__1__4435___data *)__out_argv) -> maxPos_p);
  long long *nEle = (long long *)(((struct OUT__1__4435___data *)__out_argv) -> nEle_p);
  long long *si = (long long *)(((struct OUT__1__4435___data *)__out_argv) -> si_p);
  long long *sj = (long long *)(((struct OUT__1__4435___data *)__out_argv) -> sj_p);
  long long _p_j;
  long p_index_;
  long p_lower_;
  long p_upper_;
  XOMP_loop_default((long long )0, *nEle - 1,1,&p_lower_,&p_upper_);
  for (p_index_ = p_lower_; p_index_ <= p_upper_; p_index_ += 1) 
// going upwards : anti-diagnol direction
{
// going up vertically
    long long ai =  *si - p_index_;
//  going right in horizontal
    long long aj =  *sj + p_index_;
// a critical section is used inside
    similarityScore(ai,aj, *H, *P,&( *maxPos));
  }
  XOMP_barrier();
}
