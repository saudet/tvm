# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile Example to deploy TVM modules.
CXX=/opt/nec/ve/bin/nc++
LLVM_ROOT=../../../llvm-project
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=gnu++14 -O2 -fPIC -fopenmp\
	-DTVM_THREADPOOL_USE_OPENMP=1\
	-I/opt/nec/ve/nlc/2.1.0/include/\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = ${LLVM_ROOT}/compiler-rt/lib/builtins/ve/grow_stack.S\
	-L/opt/nec/ve/nlc/2.1.0/lib/ -lcblas -lblas_openmp\
	-Wl,-rpath,/opt/nec/ve/nlc/2.1.0/lib/\
	-L${TVM_ROOT}/build -ldl -pthread -rdynamic

.PHONY: clean all

all: lib/libtvm_runtime_pack.so

# Build rule for all in one TVM package library (shared)
lib/libtvm_runtime_pack.so: lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) -shared $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Build rule for all in one TVM package library
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

clean:
	rm -rf lib
