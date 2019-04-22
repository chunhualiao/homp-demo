#/bin/bash


for var in 8 16 24 36 48 56 64 72; do

  OMP_NUM_THREADS=$var
  export OMP_NUM_THREADS

  echo //////////////////////////
  echo set OpenMP threads to be  "$var"
  echo //////////////////////////

  ./testadaptive_sparse.sh 1 1000000000 > 2k_cpu_p$var.csv
done
