all: a.out a5120.out axpy.out mm.out v6.5.out v7.out testv7.out
# default size 512
a.out: rose_jacobi-acc-opt2.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

a5120.out: rose_jacobi-ompacc-opt2-5120.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

axpy.out: rose_axpy_ompacc.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

mm.out: rose_matrixmultiply-ompacc.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

v6.5.out: rose_omp_smithW-v6.5-target-inlined-work.cu xomp_cuda_lib.cu
	nvcc $^ -g -G -o $@

omp_smithW-v7.1-adaptive.o: omp_smithW-v7.1-adaptive.c
	gcc -fopenmp -g -c $^

omp_smithW-v7.1-test.o: omp_smithW-v7.1-test.c
	gcc -fopenmp -g -c $^

omp_smithW-v8-test.o: omp_smithW-v8-test.c
	gcc -fopenmp -g -c $^

v7.out: omp_smithW-v7.1-adaptive.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o v7.out -lgomp omp_smithW-v7.1-adaptive.o

omp_smithW-v8-adaptive.o: omp_smithW-v8-adaptive.c
	gcc -fopenmp -g -c $^

sw_mem_test.o : sw_mem_test.c
	gcc -fopenmp -g -c $^

v8.out: omp_smithW-v8-adaptive.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o v8.out -lgomp omp_smithW-v8-adaptive.o

testv7.out: omp_smithW-v7.1-test.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o testv7.out -lgomp omp_smithW-v7.1-test.o

testv8.out: omp_smithW-v8-test.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o testv8.out -lgomp omp_smithW-v8-test.o

memtest.out: sw_mem_test.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o memtest.out -lgomp sw_mem_test.o

clean:
	rm -rf *.out *.o
