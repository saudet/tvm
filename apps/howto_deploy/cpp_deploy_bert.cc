/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>

int main(void) {
  int batch = 1;
  int seq_length = 128;
  int n = 100;

  LOG(INFO) << "Running graph runtime, batch = " << batch << ", seq_length = " << seq_length << ", n = " << n << " times...";
  // load in the library
  DLContext ctx{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/libbertve.so");
  // create the graph runtime module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray inputs = tvm::runtime::NDArray::Empty({batch, seq_length}, DLDataType{kDLFloat, 32, 1}, ctx);
  tvm::runtime::NDArray token_types = tvm::runtime::NDArray::Empty({batch, seq_length}, DLDataType{kDLFloat, 32, 1}, ctx);
  tvm::runtime::NDArray valid_length = tvm::runtime::NDArray::Empty({batch}, DLDataType{kDLFloat, 32, 1}, ctx);
  tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({batch, 2}, DLDataType{kDLFloat, 32, 1}, ctx);

  for (int i = 0; i < batch; ++i) {
    static_cast<float*>(valid_length->data)[i] = seq_length;
  }

  // set the right input
  set_input("data0", inputs);
  set_input("data1", token_types);
  set_input("data2", valid_length);

  // run the code
  for (int i = 0; i < n; i++) {
    run();
  }

  // get the output
  get_output(0, output);
  LOG(INFO) <<  static_cast<float*>(output->data)[0] << " " << static_cast<float*>(output->data)[1];
  return 0;
}
