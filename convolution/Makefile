C_GCC = gcc
C_CLANG = clang
CFLAGS = -std=c99 `pkg-config --cflags --libs opencv`
OPENCL = -framework OpenCL  


all: gcc  
	./convolve

clang: convolve.c
	$(C_CLANG) $(CFLAGS) $(OPENCL) -o convolve convolve.c  


gcc: convolve.c 
	$(C_GCC) $(CFLAGS) $(OPENCL) -o convolve convolve.c  


run: 
	./convolve



clean:
	rm -f convolve 
