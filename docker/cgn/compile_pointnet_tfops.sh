#!/bin/bash
set -e

# 强制使用 cgn_venv 里的 Python 3.12 + TF 2.21
PYTHON=/home/stouching/Desktop/ASPIRE/external/cgn_venv/bin/python
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found"
    exit 1
fi

# 让 TF 运行时能找到 pip 安装的 nvidia CUDA 库
NVIDIA_BASE=$($PYTHON -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
LD_DIRS=$(find "$NVIDIA_BASE" -maxdepth 2 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH="${LD_DIRS}${LD_LIBRARY_PATH:-}"

CUDA_INCLUDE=' -I/usr/local/cuda/include/'
CUDA_LIB=' -L/usr/local/cuda/lib64/'
TF_CFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# 同步 TF 2.21 的 C++ 标准，并显式启用 CXX11 ABI
CXX_STD="c++17"
NVCC_GENCODE="-gencode arch=compute_120,code=sm_120"
COMMON_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64"
# 静态链接 cudart，避免 TF 进程里同时出现 cudart 12（pip）和 cudart 13（系统 nvcc 默认）
CUDART_FLAG="-cudart=static"
# g++ 手动链接阶段：用系统 CUDA 的 static cudart 替代动态 -lcudart
CUDART_STATIC="/usr/local/cuda/lib64/libcudart_static.a"

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
 tf_sampling_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC ${CUDART_STATIC} ${TF_LFLAGS} ${CUDA_LIB}

echo 'sampling compiled'

cd ../grouping

nvcc -std=${CXX_STD} ${NVCC_GENCODE} ${CUDART_FLAG} -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
 ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=${CXX_STD} -shared -o tf_grouping_so.so tf_grouping.cpp \
 tf_grouping_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC ${CUDART_STATIC} ${TF_LFLAGS} ${CUDA_LIB}

echo 'grouping compiled'

cd ../3d_interpolation

g++ -std=${CXX_STD} -shared -o tf_interpolate_so.so tf_interpolate.cpp \
 ${CUDA_INCLUDE} ${TF_CFLAGS} ${COMMON_FLAGS} -fPIC ${CUDART_STATIC} ${TF_LFLAGS} ${CUDA_LIB} -O2

echo 'interpolation compiled'

echo 'All pointnet2 TF ops compiled successfully.'
