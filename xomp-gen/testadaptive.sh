#!/bin/bash
for i in {100..80000..100}
  do 
    ./v8_gm.out 2000 $i $1 $2
done
for i in {81000..200000..1000}
  do 
    ./v8_gm.out 2000 $i $1 $2
done
