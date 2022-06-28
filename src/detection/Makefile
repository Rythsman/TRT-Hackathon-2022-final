CUDA_PATH   = /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
TRT_PATH    = /usr/lib/x86_64-linux-gnu
SM          = 86
CCFLAG      = -std=c++14 -O3 -gencode=arch=compute_$(SM),code=sm_$(SM)
SOFLAG      = $(CCFLAG)
CUDA_HOME   = /usr/local/cuda
TRT_HOME    = /usr/local/TensorRT-8.4.1.4
OPENCV_INC  = /usr/include/opencv4
OPENCV_LIB  = /usr/lib/x86_64-linux-gnu

INCLUDE     = -I. -I$(CUDA_HOME)/include/ -I$(OPENCV_INC) -I$(TRT_HOME)/include/
LDFLAG      = -lz -L$(OPENCV_LIB) -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcudnn -L$(TRT_HOME)/lib -lnvinfer -lnvonnxparser -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs


# SOURCE_CU   = $(shell find . -regextype posix-extended -regex '.*\.(cu|cuh)')
SOURCE_CU   = $(shell find . -name '*.cu')
SOURCE_CPP  = $(shell find . -name '*.cpp')
# SOURCE_PY   = $(shell find . -name '*.py')
CU_OBJ      = $(SOURCE_CU:.cu=.cu.o)
CPP_OBJ     = $(SOURCE_CPP:.cpp=.cpp.o)
# CU_DEP      = $(CU_OBJ:.cu.o=.cu.d)
# CPP_DEP     = $(CPP_OBJ:.cpp.o=.cpp.d)

# debug:
# 	@echo $(LDFLAG)

# ifneq ($(MAKECMDGOALS), clean)
# -include $(CU_DEP) $(CPP_DEP)
# endif

# $(NVCC) $(SOFLAG) -o $@.so $^ $(LDFLAG)
all: $(CU_OBJ) $(CPP_OBJ)
	$(NVCC) $(SOFLAG) -o ./mobileVitDet $^ $(LDFLAG)
# $(SOURCE_CU:%.cu=%.so)

# LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CUDA_HOME)/lib64:$(TRT_HOME)/lib:$(OPENCV_LIB) ./mobileVitDet
# @echo "test Python code"
# python3 test.py

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
	rm -rf ./*.d ./*.o ./*.so ./*.plan ./*.bin ./*.npy ./*.npz ./mobileVitDet

