#!/bin/bash
set -e

# 容器内版本：使用系统 python，无需静态 cudart hack
PYTHON=/usr/bin/python

CUDA_INCLUDE=' -I/usr/local/cuda/include/'
CUDA_LIB=' -L/usr/local/cuda/lib64/'
TF_CFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# C++17 标准，启用 CXX11 ABI
CXX_STD="c++17"
NVCC_GENCODE="-gencode arch=compute_120,code=sm_120"
COMMON_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64"

# 容器内 CUDA 和 TF 版本匹配，使用动态链接 cudart
CUDART_FLAG="-cudart=shared"

echo "Python: $PYTHON"
echo "TF version: $($PYTHON -c 'import tensorflow as tf; print(tf.__version__)')"
echo "TF_CFLAGS: $TF_CFLAGS"
echo "TF_LFLAGS: $TF_LFLAGS"
echo "NVCC_GENCODE: $NVCC_GENCODE"
echo "CUDART_FLAG: $CUDART_FLAG"

cd pointnet2/tf_ops/sampling

nvcc -std=${CXX_STD} ${NVCC_GENCODE} ${CUDART_FLAG} -c -o tf_sampling_g.cu.o tf_sampling_g.cu \
 ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=${CXX_STD} -shared -o tf_sampling_so.so tf_sampling.cpp \
 tf_sampling_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

echo 'sampling compiled'

cd ../grouping

nvcc -std=${CXX_STD} ${NVCC_GENCODE} ${CUDART_FLAG} -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
 ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=${CXX_STD} -shared -o tf_grouping_so.so tf_grouping.cpp \
 tf_grouping_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

echo 'grouping compiled'

cd ../3d_interpolation

g++ -std=${CXX_STD} -shared -o tf_interpolate_so.so tf_interpolate.cpp \
 ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB} -O2

echo 'interpolation compiled'

echo 'All pointnet2 TF ops compiled successfully in container.'
