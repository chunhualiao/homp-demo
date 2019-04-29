all: a.out a5120.out axpy.out mm.out v8.3_gm.out
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

v7.out: omp_smithW-v7.1-adaptive.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o v7.out -lgomp omp_smithW-v7.1-adaptive.o

omp_smithW-v8.2-adaptive-single-nEle.o: omp_smithW-v8.2-adaptive-single-nEle.c
	gcc -fopenmp -g -c $^

omp_smithW-v8.3-adaptive-global-mem.o: omp_smithW-v8.3-adaptive-global-mem.c
	gcc -fopenmp -g -c $^

omp_smithW-v8.3.2-adaptive-global-mem.o: omp_smithW-v8.3.2-adaptive-global-mem.c
	gcc -fopenmp -g -c $^

omp_smithW-v8.3.4-adaptive-global-mem-num_threads.o: omp_smithW-v8.3.4-adaptive-global-mem-num_threads.c
	gcc -fopenmp -g -c $^

omp_smithW-v8-adaptive-unified-mem-outlined.o: omp_smithW-v8-adaptive-unified-mem-outlined.c
	clang -fopenmp -g -c $^
	#gcc -fopenmp -g -c $^

omp_smithW-v8-adaptive-unified-mem-outlined.o: omp_smithW-v8-adaptive-unified-mem-outlined.c
	gcc -fopenmp -g -c $^

v8.2_single.out: omp_smithW-v8.2-adaptive-single-nEle.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.2_single.out -lgomp omp_smithW-v8.2-adaptive-single-nEle.o

v8.3_gm.out: omp_smithW-v8.3-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3_gm.out -lgomp omp_smithW-v8.3-adaptive-global-mem.o

v8.3.2_gm.out: omp_smithW-v8.3.2-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3.2_gm.out -lgomp omp_smithW-v8.3.2-adaptive-global-mem.o

v8.3.4_gm.out: omp_smithW-v8.3.4-adaptive-global-mem-num_threads.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3.4_gm.out -lgomp omp_smithW-v8.3.4-adaptive-global-mem-num_threads.o

v8_um.out: omp_smithW-v8-adaptive-unified-mem-outlined.o xomp_cuda_lib.cu omp_smithW-v8-adaptive-unified-mem.cu
	nvcc -g -G -std=c++11 omp_smithW-v8-adaptive-unified-mem.cu -o v8_um.out omp_smithW-v8-adaptive-unified-mem-outlined.o -L/opt/llvm/clang+llvm-7.0.0/lib -lomp

clean:
	rm -rf *.out *.o
