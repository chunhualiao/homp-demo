all: a.out a5120.out axpy.out mm.out v8.3_gm.out
NVCC_FLAGS=-O3 -DSKIP_BACKTRACK -g

# default size 512
a.out: rose_jacobi-acc-opt2.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

a5120.out: rose_jacobi-ompacc-opt2-5120.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

axpy.out: rose_axpy_ompacc.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

mm.out: rose_matrixmultiply-ompacc.cu xomp_cuda_lib.cu
	nvcc $^ -o $@

# v6: baseline target version
# ----------------------------------
#  data map is before inner loop
v6.5.out: rose_omp_smithW-v6.5-target-inlined-work.cu xomp_cuda_lib.cu
	nvcc $(NVCC_FLAGS) $^ -o $@
# data map is before the outer loop: optimzed version
v6.6.out: rose_omp_smithW-v6.6-target-inlined-opt.cu xomp_cuda_lib.cu
	nvcc $(NVCC_FLAGS) $^ -o $@
# v7: adaptive
# ----------------------------------
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

#omp_smithW-v8-adaptive-unified-mem-outlined.o: omp_smithW-v8-adaptive-unified-mem-outlined.c
#	gcc -fopenmp -g -c $^

omp_smithW-v9-adaptive-smart-transfer-sync.o: omp_smithW-v9-adaptive-smart-transfer-sync.c
	gcc -fopenmp -g -c $^

omp_smithW-v9-adaptive-smart-transfer-async.o: omp_smithW-v9-adaptive-smart-transfer-async.c
	gcc -fopenmp -g -c $^

v8.2_single.out: omp_smithW-v8.2-adaptive-single-nEle.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.2_single.out -lgomp omp_smithW-v8.2-adaptive-single-nEle.o

sw_s1.out: sw_serial_case1.c
	gcc -fopenmp $^ -o sw_s1.out

sw_p1.out: sw_cpu_parallel_case1.c
	gcc -fopenmp $^ -o sw_p1.out
 
sw_g1.o: sw_gpu_case1.c
	gcc -fopenmp -g -c $^ -o sw_g1.o
 
sw_g1.out: sw_g1.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o sw_g1.out -lgomp sw_g1.o

sw_adaptive_gm_case1.o: sw_adaptive_gm_case1.c
	gcc -fopenmp -g -c $^

sw_a1.out: sw_adaptive_gm_case1.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o sw_a1.out -lgomp sw_adaptive_gm_case1.o

v8.3_gm.out: omp_smithW-v8.3-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3_gm.out -lgomp omp_smithW-v8.3-adaptive-global-mem.o

v8.3.2_gm.out: omp_smithW-v8.3.2-adaptive-global-mem.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3.2_gm.out -lgomp omp_smithW-v8.3.2-adaptive-global-mem.o

v8.3.4_gm.out: omp_smithW-v8.3.4-adaptive-global-mem-num_threads.o omp_smithW-v8-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v8-adaptive-gm-func.cu -o v8.3.4_gm.out -lgomp omp_smithW-v8.3.4-adaptive-global-mem-num_threads.o

v8_um.out: omp_smithW-v8-adaptive-unified-mem-outlined.o xomp_cuda_lib.cu omp_smithW-v8-adaptive-unified-mem.cu
	nvcc -g -G -std=c++11 omp_smithW-v8-adaptive-unified-mem.cu -o v8_um.out omp_smithW-v8-adaptive-unified-mem-outlined.o -L/opt/llvm/clang+llvm-7.0.0/lib -lomp

v9s.out: omp_smithW-v9-adaptive-smart-transfer-sync.o omp_smithW-v9-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v9-adaptive-gm-func.cu -o v9s.out -lgomp omp_smithW-v9-adaptive-smart-transfer-sync.o

v9as.out: omp_smithW-v9-adaptive-smart-transfer-async.o omp_smithW-v9-adaptive-gm-func.cu xomp_cuda_lib.cu
	nvcc xomp_cuda_lib.cu -g -G omp_smithW-v9-adaptive-gm-func.cu -o v9as.out -lgomp omp_smithW-v9-adaptive-smart-transfer-async.o

clean:
	rm -rf *.out *.o
