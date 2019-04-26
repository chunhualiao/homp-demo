#!/bin/bash
for i in {100..2100..100}
  do 
    ./v8.2_single.out 2000 20000 1 $i
done
