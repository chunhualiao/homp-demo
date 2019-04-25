all: a.out a5120.out axpy.out mm.out v8_gm.out v8_um.out
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

omp_smithW-v8.4-adaptive-global-mem.o: omp_smithW-v8.4-adaptive-global-mem.c
	gcc -fopenmp -g -c $^

omp_smithW-v8-adaptive-single-nEle.o: omp_smithW-v8-adaptive-single-nEle.c
	gcc -fopenmp -g -c $^

omp_smithW-v8-adaptive-global-mem.o: omp_smithW-v8-adaptive-global-mem.c
	gcc -fopenmp -g -c $^

omp_smithW-v8-adaptive-unified-mem-outlined.o: omp_smithW-v8-adaptive-unified-mem-outlined.c
	gcc -fopenmp -g -c $^

sw_mem_test.o : sw_mem_test.c
	gcc -fopenmp -g -c $^

v8_single.out: omp_smithW-v8-adaptive-single-nEle.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8_single.out -lgomp omp_smithW-v8-adaptive-single-nEle.o

v8_gm.out: omp_smithW-v8-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8_gm.out -lgomp omp_smithW-v8-adaptive-global-mem.o

v8.4_gm.out: omp_smithW-v8.4-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.4_gm.out -lgomp omp_smithW-v8.4-adaptive-global-mem.o

v8_um.out: omp_smithW-v8-adaptive-unified-mem-outlined.o xomp_cuda_lib.cu omp_smithW-v8-adaptive-unified-mem.cu
	nvcc -g -G omp_smithW-v8-adaptive-unified-mem.cu -o v8_um.out -lgomp omp_smithW-v8-adaptive-unified-mem-outlined.o

memtest.out: sw_mem_test.o rose_omp_smithW-v7.1-adaptive-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G rose_omp_smithW-v7.1-adaptive-func.cu -o memtest.out -lgomp sw_mem_test.o

clean:
	rm -rf *.out *.o
