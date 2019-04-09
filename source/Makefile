
CCFLAGS=--std=c++11

all: coo.o csf.o dense.o hicoo.o

coo.o: coo.cu
	nvcc $(CCFLAGS) -o coo.o coo.cu

csf.o: csf.cu
	nvcc $(CCFLAGS) -o csf.o csf.cu

dense.o: dense.cu
	nvcc $(CCFLAGS) -o dense.o dense.cu

hicoo.o: hicoo.cu
	nvcc $(CCFLAGS) -o hicoo.o hicoo.cu

