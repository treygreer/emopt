UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
# Linux
	CC = g++
	SHARED_FLAG = -shared
else
	ifeq ($(UNAME), Darwin)
# OS X
		CC = g++
		SHARED_FLAG = -dynamiclib
	else
		CC = g++
	endif
endif

CXXFLAGS = -Wall -fopenmp -march=native -std=c++11 -g -DNDEBUG -O3
NVCXXFLAGS = -std=c++11 -lineinfo -O3 -DNDEBUG # -Xptxas="-v"

INCLUDES = -I${EIGEN_DIR} -I${BOOST_DIR}
LDFLAGS = -lpthread -lcuda -lcudart -L/usr/local/cuda/lib64 -g

all: Grid.o Grid_ctypes.o FDTD.o Grid_cuda.o Grid_cuda_ctypes.o FDTD_cuda.o 
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o Grid.so Grid.o Grid_ctypes.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD.so FDTD.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o Grid_cuda.so Grid_cuda.o Grid_cuda_ctypes.o ${LDFLAGS}
	${CC} ${SHARED_FLAG} -fopenmp -fPIC -o FDTD_cuda.so FDTD_cuda.o ${LDFLAGS}
	cp Grid.so ../emopt
	cp FDTD.so ../emopt
	cp Grid_cuda.so ../emopt
	cp FDTD_cuda.so ../emopt

Grid.o: Grid.cpp Grid.hpp
	${CC} -c -fPIC Grid.cpp ${CXXFLAGS} -o Grid.o ${INCLUDES}

Grid_ctypes.o: Grid.cpp Grid.hpp Grid_ctypes.cpp Grid_ctypes.hpp
	${CC} -c -fPIC Grid_ctypes.cpp ${CXXFLAGS} -o Grid_ctypes.o ${INCLUDES}

FDTD.o: fdtd.cpp fdtd.hpp
	${CC} -c -fPIC fdtd.cpp ${CXXFLAGS} -o FDTD.o

Grid_cuda.o: Grid_cuda.cpp Grid_cuda.hpp
	${CC} -c -fPIC Grid_cuda.cpp ${CXXFLAGS} -o Grid_cuda.o ${INCLUDES}
##  nvcc -c --compiler-options '-fPIC -Wall' Grid_cuda.cu ${NVCXXFLAGS} -o Grid_cuda.o ${INCLUDES}

Grid_cuda_ctypes.o: Grid_cuda.cpp Grid_cuda.hpp Grid_cuda_ctypes.cpp Grid_cuda_ctypes.hpp
	${CC} -c -fPIC Grid_cuda_ctypes.cpp ${CXXFLAGS} -o Grid_cuda_ctypes.o ${INCLUDES}

FDTD_cuda.o: fdtd_cuda.cu fdtd_cuda.hpp
	nvcc -c --compiler-options '-fPIC -Wall' fdtd_cuda.cu ${NVCXXFLAGS} -o FDTD_cuda.o

clean:
	rm -f ./*.o
	rm -f ./*.so
	rm -f ../emopt/*.so
	rm -f ../emopt/*.pyc

