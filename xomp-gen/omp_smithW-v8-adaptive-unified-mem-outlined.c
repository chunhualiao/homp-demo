
#ifdef __cplusplus
extern "C" {
#endif
int MATCH_SCORE = 3;
int MISSMATCH_SCORE = -3;
int GAP_SCORE = 2;
typedef unsigned long long maxpos_t;

/*--------------------------------------------------------------------
 * Constants
 */
#define PATH -1
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3
/* End of constants */


void similarityScore_ompparallel_outlined(long long int, long long int, long long int, int*, int*, maxpos_t*, char*, char*, long long int, const int, const int, const int);


void similarityScore_ompparallel(long long int i, long long int j, int *H, int *P, maxpos_t *maxPos, char *a, char *b,
                                 long long int m) {

    int up, left, diag;

    //Stores index of element
    long long int index = m * i + j;

    //Get element above
    up = H[index - m] + GAP_SCORE;

    //Get element on the left
    left = H[index - 1] + GAP_SCORE;

    //Get element on the diagonal
    int t_mms;

    if (a[j - 1] == b[i - 1])
        t_mms = MATCH_SCORE;
    else
        t_mms = MISSMATCH_SCORE;

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
    if (max > H[*maxPos]) {
#pragma omp critical
        *maxPos = index;
    }
}

void similarityScore_ompparallel_outlined(long long int si, long long int sj,
                                          long long int nEle, int *H, int *P, maxpos_t *maxPos, char *a, char *b,
                                          long long int m, const int _MATCH_SCORE, const int _MISSMATCH_SCORE,
                                          const int _GAP_SCORE) {


    MATCH_SCORE = _MATCH_SCORE;
    MISSMATCH_SCORE = _MISSMATCH_SCORE;
    GAP_SCORE = _GAP_SCORE;
    int j;
    #pragma omp parallel for private(j)
    for (j = 0; j < nEle; ++j) {  // going upwards : anti-diagnol direction
        long long int ai = si - j; // going up vertically
        long long int aj = sj + j;  //  going right in horizontal
        similarityScore_ompparallel(ai, aj, H, P, maxPos, a, b, m); // a critical section is used inside
    }
}

#ifdef __cplusplus
}
#endif

