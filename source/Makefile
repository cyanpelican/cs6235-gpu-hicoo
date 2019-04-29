
# enable -DNDEBUG to turn off asserts for official timing
#CCFLAGS := --std=c++11  -g # -DNDEBUG
CCFLAGS := --std=c++11  -g -DNDEBUG

all: HiCooExperiment main.o coo.o dense.o hicoo.o

HiCooExperiment: main.o coo.o dense.o hicoo.o
	nvcc $(CCFLAGS) -o HiCooExperiment main.o coo.o dense.o hicoo.o

main.o: main.cu coo.hpp dense.hpp hicoo.hpp common.hpp
	nvcc $(CCFLAGS) -c -o main.o main.cu

coo.o: coo.cu coo.hpp common.hpp dense.hpp hicoo.hpp
	nvcc $(CCFLAGS) -c -o coo.o coo.cu

dense.o: dense.cu dense.hpp coo.hpp common.hpp
	nvcc $(CCFLAGS) -c -o dense.o dense.cu

hicoo.o: hicoo.cu hicoo.hpp coo.hpp common.hpp dense.hpp
	nvcc $(CCFLAGS) -c -o hicoo.o hicoo.cu


datasets: get-datasets.sh
	./get-datasets.sh


.PHONY: clean
clean:
	rm *.o HiCooExperiment
