#!/bin/bash
for i in {100..4000..100}
  do 
    ./v8.3_gm.out 2000 $i $1 $2
done
for i in {8000..200000..4000}
  do 
    ./v8.3_gm.out 2000 $i $1 $2
done
