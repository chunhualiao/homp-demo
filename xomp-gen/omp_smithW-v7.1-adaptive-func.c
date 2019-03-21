
#define NONE 0
#define UP 1
#define LEFT 2
#define DIAGONAL 3


void calculate(char* a, char* b, long long int nEle, long long int m, long long int n, int gapScore, int matchScore, int missmatchScore, long long int si, long long int sj, int* H, int* P, long long int* maxPos_ptr, long long int j, int asz) {
#pragma omp target map (to:a[0:m], b[0:n], nEle, m,n,gapScore, matchScore, missmatchScore, si, sj) map(tofrom: H[0:asz], P[0:asz], maxPos_ptr[0:1])
#pragma omp parallel for default(none) private(j) shared (a,b, nEle, m, n, gapScore, matchScore, missmatchScore, si, sj, H, P, maxPos_ptr)
          for (j = 0; j < nEle; ++j) 
	  {  // going upwards : anti-diagnol direction
	    long long int ai = si - j ; // going up vertically
	    long long int aj = sj + j;  //  going right in horizontal
	    ///------------inlined ------------------------------------------
	    //            similarityScore(ai, aj, H, P, &maxPos); // a critical section is used inside
	    {
	      int up, left, diag;

	      //Stores index of element
	      long long int index = m * ai + aj;

	      //Get element above
	      up = H[index - m] + gapScore;

	      //Get element on the left
	      left = H[index - 1] + gapScore;

	      //Get element on the diagonal
	      int t_mms;

	      if (a[aj - 1] == b[ai - 1])
		t_mms = matchScore;
	      else
		t_mms = missmatchScore;

	      diag = H[index - m - 1] + t_mms; // matchMissmatchScore(i, j);

	      // degug here
	      // return;
	      //Calculates the maximum
	      int max = NONE;
	      int pred = NONE;
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

	    }
	    // ---------------------------------------------------------------
	  }
        }
//      } // for end nDiag
//    } // end omp parallel
//}
