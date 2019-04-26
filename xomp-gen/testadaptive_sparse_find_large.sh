#!/bin/bash
for i in {1000..46000..1000}
  do 
    ./v8.2_single.out 45000 45000 1 $i
done
