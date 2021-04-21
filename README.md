# TVM for NEC SX-Aurora VE

This repository is a clone of public TVM repository
(https://github.com/apache/tvm), plus experimental modifications
which provide support for the NEC SX-Aurora TSUBASA Vector Engine (VE).

## How to build

 * After installing all tools necessary for LLVM-VE and TVM, run:
    ```bash
    git clone https://github.com/sx-aurora-dev/llvm-project/
    git clone https://github.com/sx-aurora-dev/vednn
    mkdir llvm-project/build
    cd llvm-project/build
    cmake -DCMAKE_BUILD_TYPE=Release ../llvm
    make -j100
    cd ../..

    git clone https://github.com/saudet/tvm
    mkdir tvm/build
    cd tvm/build
    git checkout aurora
    git submodule update --init --recursive
    cmake -DBUILD_FOR_VE=TRUE -DUSE_LLVM=../../llvm-project/build/bin/llvm-config ..
    make -j100
    cd ../..

    git clone https://github.com/siju-samuel/darknet
    cd darknet
    git checkout tvm
    make
    mkdir -p ~/.tvm_test_data/darknet
    cp libdarknet.so ~/.tvm_test_data/darknet/libdarknet2.0.so
    cd ..
    ```

## How to deploy

 * BERT model from [Speed up your BERT inference by 3x on CPUs using Apache TVM](https://medium.com/apache-mxnet/speed-up-your-bert-inference-by-3x-on-cpus-using-apache-tvm-9cf7776cd7f8):
    ```bash
    export TVM_HOME=$(pwd)/tvm/
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    cd tvm/apps/howto_deploy
    make
    lib/cpp_deploy_normal #
    lib/cpp_deploy_pack # small functions just to test
    OMP_NUM_THREADS=8 lib/cpp_deploy_bert
    ```

 * Benchmark for "resnet-50", "mobilenet", "vgg-19", "inception_v3", etc ImageNet models:
    ```bash
    export TVM_HOME=$(pwd)/tvm/
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    cd tvm/apps/benchmark
    make
    python3 ve_imagenet_bench.py
    ```

 * Darknet models like YOLOv2, YOLOv3, etc
    ```
    export TVM_HOME=$(pwd)/tvm/
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    cd tvm/tutorials/frontend
    make
    python3 from_darknet_on_ve.py
    ```

**Notes**: 
 * The `libtvm_runtime.so` created via CMake won't work with, for example, `cpp_deploy_normal`
   * This is because it's for the x86 host, not the VE target (use a `libtvm_runtime_pack.so` for that)
 * The functions from a static "TVM system lib" do not get registered with TVM properly for some reason
   * Instead, we need to export models to shared libraries
 * The deployed libraries link and run well with BLAS from NLC and OpenMP from NCC
   * However, more work needs to be done to link with oneDNN and veDNN
 * The vectorized code currently generated by LLVM-VE for VPU crashes with `Segmentation fault`
   * Please refer to https://github.com/sx-aurora-dev/llvm-project/issues/24

<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

Apache TVM (incubating) is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Checkout the [Contributor Guide](https://tvm.apache.org/docs/contribute/)

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
  originates from Halide. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
