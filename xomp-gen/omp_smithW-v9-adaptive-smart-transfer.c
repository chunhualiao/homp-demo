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
#include "libxomp.h"

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

double time_stamp()
{
 struct timeval t;
 double time;
 gettimeofday(&t, NULL);
 time = t.tv_sec + 1.0e-6*t.tv_usec;
 return time;
}

double omp_get_wtime()
{
  return time_stamp();
}

#endif
extern void calculate(char*, char*, long long int, long long int, long long int, int, int, int, long long int, long long int, int*, int*, long long int*, long long int, int, int, int);
/*--------------------------------------------------------------------
 * Functions Prototypes
 */
//#pragma omp declare target

//Defines size of strings to be compared
long long int m = 8 ; //Columns - Size of string a
long long int n = 9;  //Lines - Size of string b
int gapScore = -2;

//Defines scores
int matchScore = 3;
int missmatchScore = -3;

//Strings over the Alphabet Sigma
char *a, *b;

int matchMissmatchScore(long long int i, long long int j);
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos);
//#pragma omp end declare target


// without omp critical: how to conditionalize it?
void similarityScore2(long long int i, long long int j, int* H, int* P, long long int* maxPos);
void backtrack(int* P, long long int maxPos);
void printMatrix(int* matrix);
void printPredecessorMatrix(int* matrix);
void generate(void);
long long int nElement(long long int i);
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj);

void recoverScore(long long int i, int* H, int* P, int GPUDataSize, int* HGPU, int* PGPU) ;
/* End of prototypes */


/*--------------------------------------------------------------------
 * Global Variables
 */
bool useBuiltInData=true;

int MEDIUM=1;
//int MEDIUM=1200;
int LARGE=1; // max 46340 for GPU of 16GB Device memory
//int LARGE=8000; // max 46340 for GPU of 16GB Device memory

// the generated scoring matrix's size is m++ and n++ later to have the first row/column as 0s.

/* End of global variables */

/*--------------------------------------------------------------------
 * Function:    main
 */
extern void xomp_acc_init();
int main(int argc, char* argv[]) {
  // thread_count is no longer used
  int thread_count;
  xomp_acc_init();
  bool GPUDataCopied = false;
  if (argc==3)
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    useBuiltInData = false;
  } else if (argc == 4) 
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    LARGE = atoi(argv[3]);
    useBuiltInData = false;
  } else if (argc == 5) 
  {
    m = strtoll(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
    MEDIUM = strtoll(argv[3], NULL, 10);
    LARGE = strtoll(argv[4], NULL, 10);
    useBuiltInData = false;
  }

//#ifdef DEBUG
  if (useBuiltInData)
  {
    printf ("Usage: %s m n\n", argv[0]);
    printf ("Using built-in data for testing ..\n");
  }
//  printf("Problem size: Matrix[%lld][%lld], Medium=%d Large=%d\n", n, m, MEDIUM, LARGE);
//#endif

    //Because now we have zeros
    m++;
    n++;

    //Allocates a and b
    a = (char*) malloc(m * sizeof(char));
//    printf ("debug: a's address=%p\n", a);

    b = (char*) malloc(n * sizeof(char));
//    printf ("debug: b's address=%p\n", b);

    //Allocates similarity matrix H
    int *H;
    H = (int *) calloc(m * n, sizeof(int));
//    printf ("debug: H's address=%p\n", H);

    //Allocates predecessor matrix P
    int *P;
    P = (int *)calloc(m * n, sizeof(int));
//    printf ("debug: P's address=%p\n", P);

    long long int GPUIterSizeSlope = 2*max(0, min(m,n)-1-LARGE);
    long long int GPUIterSizeFlat = 0;
    if (LARGE <= min(m,n)) {
        GPUIterSizeFlat = max(m,n) - min(m,n) + 1;
    };
    long long int GPUDataSize = min(m,n)*(GPUIterSizeSlope + GPUIterSizeFlat + 2);
    int *HGPU;
    HGPU = (int *) calloc(GPUDataSize, sizeof(int));
    int *PGPU;
    PGPU = (int *) calloc(GPUDataSize, sizeof(int));

    if (useBuiltInData)
    {
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
      b[0] =   'G';
      b[1] =   'G';
      b[2] =   'T';
      b[3] =   'T';
      b[4] =   'G';
      b[5] =   'A';
      b[6] =   'C';
      b[7] =   'T';
      b[8] =   'A';

      a[0] =   'T';
      a[1] =   'G';
      a[2] =   'T';
      a[3] =   'T';
      a[4] =   'A';
      a[5] =   'C';
      a[6] =   'G';
      a[7] =   'G';
    }
    else
    {
      //Gen random arrays a and b
      generate();
    }

    //Start position for backtrack
    long long int maxPos = 0;
    //Calculates the similarity matrix
    long long int i, j;


    // The way to generate all wavefront is to go through the top edge elements
    // starting from the left top of the matrix, go to the bottom top -> down, then left->right
    // total top edge element count =  dim1_size + dim2_size -1 
    //Because now we have zeros ((m-1) + (n-1) - 1)
    long long int nDiag = m + n - 3;

#ifdef DEBUG
    printf("nDiag=%lld\n", nDiag);
    printf("Number of wavefront lines and their first element positions:\n");
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

    // time measurements for each iteration of outer loop.
    double iterationTime;
    // time measurements for GPU memory copy.
    double memCopyInGPUTime;
    double memCopyOutGPUTime;

    // mistake: element count, not byte size!!
    // int asz= m*n*sizeof(int);
    int asz= m*n;
// choice 2: map data before the outer loop
//#pragma omp target map (to:a[0:m], b[0:n], nDiag, m,n,gapScore, matchScore, missmatchScore) map(tofrom: H[0:asz], P[0:asz], maxPos)
//  #pragma omp parallel default(none) shared(H, P, maxPos, nDiag, j) private(i)
	    long long int *maxPos_ptr = &maxPos;
    /*
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
*/
    {
      long long int last1i = 1;
      long long int last1j = 0;
      long long int last2i = 0;
      long long int last2j = 0;
      int GPUDataOffset = 1;
      for (i = 1; i <= nDiag; ++i) { // start from 1 since 0 is the boundary padding
          long long int nEle, si, sj;
          //  nEle = nElement(i);
          //---------------inlined ------------
          if (i < m && i < n) { // smaller than both directions
              //Number of elements in the diagonal is increasing
              nEle = i;
          } else if (i < max(m, n)) { // smaller than only one direction
              //Number of elements in the diagonal is stable
              long int min = min(m,
                                 n);  // the longer direction has the edge elements, the number is the smaller direction's size
              nEle = min - 1;
          } else {
              //Number of elements in the diagonal is decreasing
              long int min = min(m, n);
              nEle = 2 * min - i + llabs(m - n) - 2;
          }

          //calcFirstDiagElement(i, &si, &sj);
          //------------inlined---------------------
          // Calculate the first element of diagonal
          if (i < n) { // smaller than row count
              si = i;
              sj = 1; // start from the j==1 since j==0 is the padding
          } else {  // now we sweep horizontally at the bottom of the matrix
              si = n - 1;  // i is fixed
              sj = i - n + 2; // j position is the nDiag (id -n) +1 +1 // first +1
              GPUDataOffset = 0;
          }

          // Copy the data for the first time GPU will be used
          if (nEle >= LARGE && !GPUDataCopied) {
              memCopyInGPUTime = omp_get_wtime();
              // copy last two diagonals to HGPU and PGPU.
              // copy -1 diagonal
              int di, dj;
              di = last2i;
              dj = last2j;
              long long int GPUDataIndex = 0;
              int GPUCopyOffset = 0;
              if (di < n) {
                  GPUCopyOffset = 1;
              };
              while (di >= 0 && dj < m) {
                HGPU[GPUDataIndex+GPUCopyOffset] = H[di*m+dj];
                PGPU[GPUDataIndex+GPUCopyOffset] = P[di*m+dj];
                GPUDataIndex++;
                di--;
                dj++;
              };
              GPUCopyOffset = 0;
              di = last1i;
              dj = last1j;
              if (di < n) {
                  GPUCopyOffset = 1;
              };
              GPUDataIndex = min(m,n);
              while (di >= 0 && dj < m) {
                HGPU[GPUDataIndex+GPUCopyOffset] = H[di*m+dj];
                PGPU[GPUDataIndex+GPUCopyOffset] = P[di*m+dj];
                GPUDataIndex++;
                di--;
                dj++;
              };

              xomp_deviceDataEnvironmentEnter(0);
              char *_dev_a;
              int _dev_a_size[1] = {m};
              int _dev_a_offset[1] = {0};
              int _dev_a_Dim[1] = {m};
              _dev_a = ((char *) (xomp_deviceDataEnvironmentPrepareVariable(0, (void *) a, 1, sizeof(char), _dev_a_size,
                                                                            _dev_a_offset, _dev_a_Dim, 1, 0)));
              char *_dev_b;
              int _dev_b_size[1] = {n};
              int _dev_b_offset[1] = {0};
              int _dev_b_Dim[1] = {n};
              _dev_b = ((char *) (xomp_deviceDataEnvironmentPrepareVariable(0, (void *) b, 1, sizeof(char), _dev_b_size,
                                                                            _dev_b_offset, _dev_b_Dim, 1, 0)));
              int *_dev_H;
              int _dev_H_size[1] = {GPUDataSize};
              int _dev_H_offset[1] = {0};
              int _dev_H_Dim[1] = {GPUDataSize};
              _dev_H = ((int *) (xomp_deviceDataEnvironmentPrepareVariable(0, (void *) HGPU, 1, sizeof(int), _dev_H_size,
                                                                           _dev_H_offset, _dev_H_Dim, 1, 1)));
              int *_dev_P;
              int _dev_P_size[1] = {GPUDataSize};
              int _dev_P_offset[1] = {0};
              int _dev_P_Dim[1] = {GPUDataSize};
              _dev_P = ((int *) (xomp_deviceDataEnvironmentPrepareVariable(0, (void *) PGPU, 1, sizeof(int), _dev_P_size,
                                                                           _dev_P_offset, _dev_P_Dim, 1, 1)));
              long long *_dev_maxPos_ptr;
              int _dev_maxPos_ptr_size[1] = {1};
              int _dev_maxPos_ptr_offset[1] = {0};
              int _dev_maxPos_ptr_Dim[1] = {1};
              _dev_maxPos_ptr = ((long long *) (xomp_deviceDataEnvironmentPrepareVariable(0, (void *) maxPos_ptr, 1,
                                                                                          sizeof(long long),
                                                                                          _dev_maxPos_ptr_size,
                                                                                          _dev_maxPos_ptr_offset,
                                                                                          _dev_maxPos_ptr_Dim, 1, 1)));
              memCopyInGPUTime = omp_get_wtime() - memCopyInGPUTime;
              GPUDataCopied = true;
          };

          // Copy data back to CPU after the last time GPU is used
          if (GPUDataCopied && nEle < LARGE) {
              GPUDataCopied = false;
              memCopyOutGPUTime = omp_get_wtime();
              xomp_sync();
              //xomp_deviceDataEnvironmentExit(0);
              recoverScore(i-2, H, P, GPUDataSize, HGPU, PGPU);
              memCopyOutGPUTime = omp_get_wtime() - memCopyOutGPUTime;
          };

          // serial version: 0 to < medium: small data set
          if (nEle < MEDIUM) {
              iterationTime = omp_get_wtime();
              for (j = 0; j < nEle; ++j) {  // going upwards : anti-diagnol direction
                  long long int ai = si - j; // going up vertically
                  long long int aj = sj + j;  //  going right in horizontal
                  similarityScore2(ai, aj, H, P, &maxPos); // a specialized version without a critical section used inside
              }
              iterationTime = omp_get_wtime() - iterationTime;
              //printf("CPU Sequential iteration: %d, nEle: %d, time: %fs.\n", i, nEle, iterationTime);
              //printf("%d, %d, %f\n", i, nEle, iterationTime);
              //if (i%600 == 0) {
              //  printf("%d, %d, %f\n", i, nEle, iterationTime);
              //};
          } else if (nEle < LARGE) { // omp cpu version: medium to large: medium data set
              iterationTime = omp_get_wtime();
#pragma omp parallel for private(j) shared (nEle, si, sj, H, P)
              for (j = 0; j < nEle; ++j) {  // going upwards : anti-diagnol direction
                  long long int ai = si - j; // going up vertically
                  long long int aj = sj + j;  //  going right in horizontal
                  similarityScore(ai, aj, H, P, &maxPos); // a critical section is used inside
              }
              iterationTime = omp_get_wtime() - iterationTime;
              //printf("CPU Parallel iteration: %d, nEle: %d, time: %fs.\n", i, nEle, iterationTime);
              //printf("%d, %d, %f\n", i, nEle, iterationTime);
              //if (i%600 == 0) {
              //  printf("%d, %d, %f\n", i, nEle, iterationTime);
              //};
          } else {// omp gpu version: large data set
// choice 1: map data before the inner loop
              iterationTime = omp_get_wtime();
              calculate(a, b, nEle, m, n, gapScore, matchScore, missmatchScore, si, sj, HGPU, PGPU, maxPos_ptr, j, GPUDataSize, i-LARGE+2, GPUDataOffset);
              xomp_deviceSmartDataTransfer(0, HGPU, 4*min(m,n), (i-LARGE+2)*4*min(m,n));
              xomp_deviceSmartDataTransfer(0, PGPU, 4*min(m,n), (i-LARGE+2)*4*min(m,n));
              iterationTime = omp_get_wtime() - iterationTime;
              //printf("GPU iteration: %d, nEle: %d, time: %fs.\n", i, nEle, iterationTime);
              //if (i%600 == 0) {
              //  printf("%d, %d, %f\n", i, nEle, iterationTime);
              //};
          }
          if (!GPUDataCopied) {
            last2i = last1i;
            last2j = last1j;
            last1i = si;
            last1j = sj;
          };
      }
    }

    // Copy data back to CPU after the last time GPU is used and there is no following outer loop iteration.
    if (GPUDataCopied) {
        GPUDataCopied = false;
        memCopyOutGPUTime = omp_get_wtime();
        xomp_sync();
        //xomp_deviceDataEnvironmentExit(0);
        recoverScore(i-2, H, P, GPUDataSize, HGPU, PGPU);
        memCopyOutGPUTime = omp_get_wtime() - memCopyOutGPUTime;
    };


  double finalTime = omp_get_wtime();
  //printf("GPU memory copy time Host to Device: %f\n", memCopyInGPUTime);
  //printf("GPU memory copy time Device to Host: %f\n", memCopyOutGPUTime);
  //printf("\nElapsed time for scoring matrix computation: %f\n", finalTime - initialTime);

    printf("%lld, %lld, %lld, %f, %f, %f, %f\n", m-1, n-1, LARGE, finalTime - initialTime, memCopyInGPUTime, memCopyOutGPUTime, memCopyInGPUTime + memCopyOutGPUTime);

  initialTime = omp_get_wtime();
  backtrack(P, maxPos);
  finalTime = omp_get_wtime();

  //Gets backtrack time
  finalTime = omp_get_wtime();
  //printf("Elapsed time for backtracking: %f\n", finalTime - initialTime);

#ifdef DEBUG
    printf("\nSimilarity Matrix:\n");
    printMatrix(H);

    printf("\nPredecessor Matrix:\n");
    printPredecessorMatrix(P);
#endif

    if (useBuiltInData)
    {
    /* 
      printf("Last H: %lld\n", H[n*m-1]);
      for (i = 0; i < n; i++) {
          for (j = 0; j < m; j++) {
              printf("%d ", H[i*m+j]);
          }
          printf("\n");
      }
      printf("H_GPU\n");
      for (i = 0; i < GPUDataSize/min(m,n); i++) {
          for (j = 0; j < min(m,n); j++) {
              printf("%d ", HGPU[i*min(m,n)+j]);
          }
          printf("\n");
      }
      */
      printf ("Verifying results using the builtinIn data: %s\n", (H[n*m-1]==7)?"true":"false");
      assert (H[n*m-1]==7);
    }

    //Frees similarity matrixes
    free(H);
    free(P);

    //Frees input arrays
    free(a);
    free(b);

    return 0;
}  /* End of main */

/*--------------------------------------------------------------------
 * Function:    nElement
 * Purpose:     Calculate the number of i-diagonal's elements
 * i value range 1 to nDiag.  we inclulde the upper bound value. 0 is for the padded wavefront, which is ignored.
 */
long long int nElement(long long int i) {
    if (i < m && i < n) { // smaller than both directions
        //Number of elements in the diagonal is increasing
        return i;
    }
    else if (i < max(m, n)) { // smaller than only one direction
        //Number of elements in the diagonal is stable
        long int min = min(m, n);  // the longer direction has the edge elements, the number is the smaller direction's size
        return min - 1;
    }
    else {
        //Number of elements in the diagonal is decreasing
        long int min = min(m, n);
        return 2 * min - i + llabs(m - n) - 2;
    }
}

/*--------------------------------------------------------------------
 * Function:    calcElement: expect valid i value is from 1 to nDiag. since the first one is 0 padding
 * Purpose:     Calculate the position of (si, sj)-element
 * n rows, m columns: we sweep the matrix on the left edge then bottom edge to get the wavefront
 */
void calcFirstDiagElement(long long int i, long long int *si, long long int *sj) {
    // Calculate the first element of diagonal
    if (i < n) { // smaller than row count
        *si = i;
        *sj = 1; // start from the j==1 since j==0 is the padding
    } else {  // now we sweep horizontally at the bottom of the matrix
        *si = n - 1;  // i is fixed
        *sj = i - n + 2; // j position is the nDiag (id -n) +1 +1 // first +1 
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
void recoverScore(long long int i, int* H, int* P, int GPUDataSize, int* HGPU, int* PGPU) {
    // lasti: the index i when GPU computing started.
    // lasti = i - GPUDataSize/min(m,n) + 3
    int j, k, index;
    int nEle, si, sj;
    int GPUDataOffset = 1;
    int row = 2;
    int len = GPUDataSize/min(m,n);
    int rowLength = min(m, n);
    int GPUDataIndex = GPUDataSize - 2*min(m,n);
    /*
    for (int p = 0; p < 2; p++) {
        for (int q = 0; q < min(m,n); q++) {
            printf("%lld ", HGPU[GPUDataIndex + q]);
        }
        printf("\n");
        GPUDataIndex += min(m,n);
    }
    GPUDataIndex = GPUDataIndex - 2*min(m,n);
    */


    for (j = 0; j < 2; j++) {
        if (i < m && i < n) { // smaller than both directions
            nEle = i;
        } else if (i < max(m, n)) { // smaller than only one direction
            long int min = min(m, n);  // the longer direction has the edge elements, the number is the smaller direction's size
            nEle = min - 1;
        } else {
            long int min = min(m, n);
            nEle = 2 * min - i + llabs(m - n) - 2;
        }

        if (i < n) { // smaller than row count
            si = i;
            sj = 1; // start from the j==1 since j==0 is the padding
        } else {  // now we sweep horizontally at the bottom of the matrix
            si = n - 1;  // i is fixed
            sj = i - n + 2; // j position is the nDiag (id -n) +1 +1 // first +1
            GPUDataOffset = 0;
        }
        index = si * m + sj;
        for (k = GPUDataOffset; k < nEle; k++) {
            H[index] = HGPU[GPUDataIndex + k];
            //P[index] = PGPU[GPUDataIndex + k];
            index = index - m + 1;
            //printf("%lld ", HGPU[GPUDataIndex + k]);
        };
        //printf("\n");
        i++;
        GPUDataIndex += rowLength;
    }
}




/*--------------------------------------------------------------------
 * Function:    SimilarityScore
 * Purpose:     Calculate  value of scoring matrix element H(i,j) : the maximum Similarity-Score H(i,j)
 *             int *P; the predecessor array,storing which of the three elements is picked with max value
 */
// #pragma omp declare target
void similarityScore(long long int i, long long int j, int* H, int* P, long long int* maxPos_ptr) {

    //long long int *maxPos_ptr = &maxPos;
    int up, left, diag;

    //Stores index of element
    long long int index = m * i + j;

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

    diag = H[index - m - 1] + t_mms; // matchMissmatchScore(i, j);

// degug here
// return;
    //Calculates the maximum
    int max = NONE;
    int pred = NONE;
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
    if (diag > max) { //same letter ↖
        max = diag;
        pred = DIAGONAL;
    }

    if (up > max) { //remove letter ↑
        max = up;
        pred = UP;
    }

    if (left > max) { //insert letter ←
        max = left;
        pred = LEFT;
    }
    //Inserts the value in the similarity and predecessor matrixes
    H[index] = max;
    P[index] = pred;

    //Updates maximum score to be used as seed on backtrack
    if (max > H[maxPos_ptr[0]]) {
        //#pragma omp critical
        maxPos_ptr[0] = index;
    }
}  /* End of similarityScore */

/*--------------------------------------------------------------------
 * Function:    matchMissmatchScore
 * Purpose:     Similarity function on the alphabet for match/missmatch
 */
int matchMissmatchScore(long long int i, long long int j) {
    if (a[j - 1] == b[i - 1])
        return matchScore;
    else
        return missmatchScore;
}  /* End of matchMissmatchScore */


//#pragma omp end declare target

void similarityScore2(long long int i, long long int j, int* H, int* P, long long int* maxPos) {

    int up, left, diag;

    //Stores index of element
    long long int index = m * i + j;

    //Get element above
    up = H[index - m] + gapScore;

    //Get element on the left
    left = H[index - 1] + gapScore;

    //Get element on the diagonal
    diag = H[index - m - 1] + matchMissmatchScore(i, j);

    //Calculates the maximum
    int max = NONE;
    int pred = NONE;
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

    if (diag > max) { //same letter ↖
        max = diag;
        pred = DIAGONAL;
    }

    if (up > max) { //remove letter ↑
        max = up;
        pred = UP;
    }

    if (left > max) { //insert letter ←
        max = left;
        pred = LEFT;
    }
    //Inserts the value in the similarity and predecessor matrixes
    H[index] = max;
    P[index] = pred;

    //Updates maximum score to be used as seed on backtrack
    if (max > H[*maxPos]) {
        *maxPos = index;
    }
}  /* End of similarityScore2 */


/*--------------------------------------------------------------------
 * Function:    backtrack
 * Purpose:     Modify matrix to print, path change from value to PATH
 */
void backtrack(int* P, long long int maxPos) {
    //hold maxPos value
    long long int predPos;

    size_t n = sizeof(P)/sizeof(P[0]);

    //backtrack from maxPos to startPos = 0
    do {
        if (P[maxPos] == DIAGONAL)
            predPos = maxPos - m - 1;
        else if (P[maxPos] == UP)
            predPos = maxPos - m;
        else if (P[maxPos] == LEFT)
            predPos = maxPos - 1;
        P[maxPos] *= PATH;
        maxPos = predPos;
    } while (maxPos >= 0 && maxPos < n && P[maxPos] != NONE);
}  /* End of backtrack */

/*--------------------------------------------------------------------
 * Function:    printMatrix
 * Purpose:     Print Matrix
 */
void printMatrix(int* matrix) {
    long long int i, j;
    printf("-\t-\t");
    for (j = 0; j < m-1; j++) {
    	printf("%c\t", a[j]);
    }
    printf("\n-\t");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {  
        	if (j==0 && i>0) printf("%c\t", b[i-1]);
            printf("%d\t", matrix[m * i + j]);
        }
        printf("\n");
    }

}  /* End of printMatrix */

/*--------------------------------------------------------------------
 * Function:    printPredecessorMatrix
 * Purpose:     Print predecessor matrix
 */
void printPredecessorMatrix(int* matrix) {
    long long int i, j, index;
    printf("    ");
    for (j = 0; j < m-1; j++) {
    	printf("%c ", a[j]);
    }
    printf("\n  ");
    for (i = 0; i < n; i++) { //Lines
        for (j = 0; j < m; j++) {
        	if (j==0 && i>0) printf("%c ", b[i-1]);
            index = m * i + j;
            if (matrix[index] < 0) {
                printf(BOLDRED);
                if (matrix[index] == -UP)
                    printf("↑ ");
                else if (matrix[index] == -LEFT)
                    printf("← ");
                else if (matrix[index] == -DIAGONAL)
                    printf("↖ ");
                else
                    printf("- ");
                printf(RESET);
            } else {
                if (matrix[index] == UP)
                    printf("↑ ");
                else if (matrix[index] == LEFT)
                    printf("← ");
                else if (matrix[index] == DIAGONAL)
                    printf("↖ ");
                else
                    printf("- ");
            }
        }
        printf("\n");
    }

}  /* End of printPredecessorMatrix */

/*--------------------------------------------------------------------
 * Function:    generate
 * Purpose:     Generate arrays a and b
 */
void generate() {
    //Random seed
    srand(time(NULL));

    //Generates the values of a
    long long int i;
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
} /* End of generate */


/*--------------------------------------------------------------------
 * External References:
 * http://vlab.amrita.edu/?sub=3&brch=274&sim=1433&cnt=1
 * http://pt.slideshare.net/avrilcoghlan/the-smith-waterman-algorithm
 * http://baba.sourceforge.net/
 */
