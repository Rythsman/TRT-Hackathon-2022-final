CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/local/TensorRT-8.4.1.4
SM          = 86
CCFLAG      = -std=c++14 -DNDEBUG -O3 -gencode=arch=compute_$(SM),code=sm_$(SM)
SOFLAG      = $(CCFLAG) -shared
INCLUDE     = -I. -I$(CUDA_PATH)/include -I$(TRT_PATH)/include
LDFLAG      = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcudnn -L$(TRT_PATH)/lib -lnvinfer

# SOURCE_CU   = $(shell find . -regextype posix-extended -regex '.*\.(cu|cuh)')
SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_CPP  = $(shell find . -name '*.cpp')
# SOURCE_PY   = $(shell find . -name '*.py')
CU_OBJ      = $(SOURCE_CU:.cu=.cu.o)
CPP_OBJ     = $(SOURCE_CPP:.cpp=.cpp.o)
# CU_DEP      = $(CU_OBJ:.cu.o=.cu.d)
# CPP_DEP     = $(CPP_OBJ:.cpp.o=.cpp.d)


# ifneq ($(MAKECMDGOALS), clean)
# -include $(CU_DEP) $(CPP_DEP)
# endif

# $(NVCC) $(SOFLAG) -o $@.so $^ $(LDFLAG)
all: $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) $(SOFLAG) -o swishPlugin.so $^ $(LDFLAG)
	
# $(SOURCE_CU:%.cu=%.so)

# %.so: %.cu.o
# 	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

%.cpp.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

%.cu.o: %.cu
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<
	
.PHONY: test
test:
	make clean
	make
	python $(SOURCE_PY)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.plan

