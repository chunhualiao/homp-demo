#!/bin/bash
# M=2000, N=20000-200000. This is the portion good for gpu calculation.
for i in {20000..200000..4000}
  do 
    ./sw_g1.out 2000 $i $1 $2
done
