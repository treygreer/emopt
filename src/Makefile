CC = g++
SHARED_FLAG = -shared

CXXFLAGS = -Wall -fopenmp -march=native -std=c++11 -g -O4 -DNDEBUG 

# to suppress irritating BOOST-related nvcc warnings, see
# https://stackoverflow.com/questions/14831051/how-to-disable-a-specific-nvcc-compiler-warnings and
# http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
NVCXX_SUPPRESS_WARNINGS = -Xcudafe "--diag_suppress=probable_guiding_friend"

NVCXXFLAGS = --compiler-options '-fPIC -Wall' -std=c++11 -lineinfo ${NVCXX_SUPPRESS_WARNINGS} -O3 -DNDEBUG  # -Xptxas="-v"

INCLUDES = -I${EIGEN_DIR} -I${BOOST_DIR} -I/usr/local/cuda/include
LDFLAGS = -lpthread -lcuda -lcudart -L/usr/local/cuda/lib64 -g

all: Grid_cuda.o Grid_cuda_ctypes.o FDTD_cuda.o CudaClipper.o
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o Grid_cuda.so Grid_cuda.o Grid_cuda_ctypes.o CudaClipper.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD_cuda.so FDTD_cuda.o ${LDFLAGS}
	cp Grid_cuda.so ../emopt
	cp FDTD_cuda.so ../emopt

Grid_cuda.o: Grid_cuda.cpp Grid_cuda.hpp
	${CC} -c -fPIC Grid_cuda.cpp ${CXXFLAGS} -o Grid_cuda.o ${INCLUDES}

CudaClipper.o: CudaClipper.cu CudaClipper.hpp Grid_cuda.hpp Makefile
#	${CC} -c -fPIC CudaClipper.cpp ${CXXFLAGS} -o CudaClipper.o ${INCLUDES}
	nvcc -c CudaClipper.cu ${NVCXXFLAGS} -I${BOOST_DIR} -o CudaClipper.o

Grid_cuda_ctypes.o: Grid_cuda.cpp Grid_cuda.hpp Grid_cuda_ctypes.cpp Grid_cuda_ctypes.hpp
	${CC} -c -fPIC Grid_cuda_ctypes.cpp ${CXXFLAGS} -o Grid_cuda_ctypes.o ${INCLUDES}

FDTD_cuda.o: fdtd_cuda.cu fdtd_cuda.hpp Makefile
	nvcc -c fdtd_cuda.cu ${NVCXXFLAGS} -o FDTD_cuda.o

clean:
	rm -f ./*.o
	rm -f ./*.so
	rm -f ../emopt/*.so
	rm -f ../emopt/*.pyc

