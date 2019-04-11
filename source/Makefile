
CCFLAGS=--std=c++11

all: HiCooExperiment main.o coo.o csf.o dense.o hicoo.o

HiCooExperiment: main.o coo.o csf.o dense.o hicoo.o
	nvcc $(CCFLAGS) -o HiCooExperiment main.o coo.o csf.o dense.o hicoo.o

main.o: main.cpp
	nvcc $(CCFLAGS) -c -o main.o main.cpp

coo.o: coo.cu coo.hpp common.hpp dense.hpp csf.hpp hicoo.hpp
	nvcc $(CCFLAGS) -c -o coo.o coo.cu

csf.o: csf.cu csf.hpp common.hpp coo.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o csf.o csf.cu

dense.o: dense.cu dense.hpp coo.hpp common.hpp
	nvcc $(CCFLAGS) -c -o dense.o dense.cu

hicoo.o: hicoo.cu hicoo.hpp coo.hpp common.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o hicoo.o hicoo.cu

.PHONY: clean
clean:
	rm *.o HiCoo
