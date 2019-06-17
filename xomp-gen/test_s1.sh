#!/bin/bash
# M=2000, N=100-1100. This is the portion good for serial calculation.
for i in {100..1100..100}
  do 
    ./sw_s1.out 2000 $i $1 $2
done
