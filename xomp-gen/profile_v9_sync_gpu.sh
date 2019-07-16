#!/bin/bash

for i in {45000..33000..-1000}
  do
    nvprof ./v9s.out 45000 45000 1 $i 2>&1 | grep " OUT\|HtoD\|DtoH" | sed 's/GPU activities://g' | awk -F" " 'BEGIN { ORS="," };{print $2}'; echo;
done
