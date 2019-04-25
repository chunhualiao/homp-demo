#!/bin/bash
for i in {1000..45000..1000}
  do 
    ./v8_single.out 45000 45000 1 $i
done
