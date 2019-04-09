
CCFLAGS=--std=c++11

all: coo.o csf.o dense.o hicoo.o

coo.o: coo.cu coo.hpp common.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o coo.o coo.cu

csf.o: csf.cu csf.hpp common.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o csf.o csf.cu

dense.o: dense.cu dense.hpp common.hpp
	nvcc $(CCFLAGS) -c -o dense.o dense.cu

hicoo.o: hicoo.cu hicoo.hpp common.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o hicoo.o hicoo.cu

clean:
	rm *.o
