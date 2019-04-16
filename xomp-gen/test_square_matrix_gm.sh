#!/bin/bash
# modify MEDIUM as $1 and LARGE as $2 to
# switch between CPU sequential, CPU parallel and GPU global memory
# e.g. ./test_square_matrix 1 1 will lead to pure GPU computation.

# For CPU performance, output would be M, N, total time.
# For GPU performance, three more columns would be mem copy time from host to device,
# device to host and the sum of two.

for i in {100..45000..100}
  do 
    ./v8_gm.out $i $i $1 $2
 done
