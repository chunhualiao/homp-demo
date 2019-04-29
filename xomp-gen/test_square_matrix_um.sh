#!/bin/bash
# modify MEDIUM as $1 and LARGE as $2 to
# switch between CPU sequential, CPU parallel and GPU unified memory
# e.g. ./test_square_matrix 1 1 will lead to pure GPU computation.

# The output would be M, N, total time.

for i in {100..45000..100}
  do 
   KMP_AFFINITY=compact numactl --cpunodebind=0 --membind=0 ./v8_um.out $i $i $1 $2
 done
