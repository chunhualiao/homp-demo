#!/bin/bash

for i in {45000..33000..-1000}
  do
    nvprof ./v9as.out 45000 45000 1 $i 2>&1 | grep " OUT\|HtoD\|DtoH\|cudaMemcpyAsync" | sed 's/API calls://g' | sed 's/GPU activities://g' | awk -F" " 'BEGIN { ORS="," };{print $2}'; echo;
done
