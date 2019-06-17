#!/bin/bash
# M=2000, N=1200-19000. This is the portion good for cpu parallel calculation.
for i in {1200..19000..1000}
  do 
    ./sw_p1.out 2000 $i $1 $2
done
