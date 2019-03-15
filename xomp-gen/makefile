all: a.out a5120.out axpy.out mm.out v6.5.out
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

clean:
	rm -rf *.out
