CXX = mpic++ # or mpicxx
CXXFLAGS = -std=c++11 -O3 -fopenmp
NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp" # pass -fopenmp to host compiler (g++)

TARGETS = $(basename $(wildcard *.cpp *.cu))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean