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

int main(int argc, char *argv[]) {
  if (argc < 5) {
    LOG(INFO) << "Usage: " << argv[0] << " [module lib] [input shape] [output shape] [repeat]";
    return 1;
  }

  char* filename = argv[1];
  std::vector<int64_t> inputShape(4), outputShape(2);
  sscanf(argv[2], "(%lld, %lld, %lld, %lld)", &inputShape[0], &inputShape[1], &inputShape[2], &inputShape[3]);
  sscanf(argv[3], "(%lld, %lld)", &outputShape[0], &outputShape[1]);
  int repeat = atoi(argv[4]);

  LOG(INFO) << "Running graph runtime...";
  // load in the library
  DLContext ctx{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(filename);
  // create the graph runtime module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  tvm::runtime::PackedFunc time_evaluator = *tvm::runtime::Registry::Get("runtime.RPCTimeEvaluator");

  // Use the C++ API
  tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(inputShape, DLDataType{kDLFloat, 32, 1}, ctx);
  tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty(outputShape, DLDataType{kDLFloat, 32, 1}, ctx);

  // set the right input
  set_input("data", input);

  // run the code
  tvm::runtime::PackedFunc ftimer = time_evaluator(gmod, "run", static_cast<int>(ctx.device_type), ctx.device_id, 1, repeat, 0, "");
  std::string rv = ftimer();

  // get the output
  get_output(0, output);

  const double* results = reinterpret_cast<const double*>(rv.data());
  for (int i = 0; i < repeat; i++) {
    std::cout << results[i] << std::endl;
  }
  return 0;
}
