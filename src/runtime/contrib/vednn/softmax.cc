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
 * \file src/runtime/contrib/vednn/softmax.cc
 * \brief Use external vednn softmax function
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <vednn.h>
vednnError_t operator |= (vednnError_t& x, vednnError_t y) { return (vednnError_t)((int&)x |= (int)y); }
#include <C/vednnInit.c>
#include <C/vednnSoftmaxForward.c>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.vednn.softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* x = args[0];
      DLTensor* y = args[1];
      int axis = args[2];
      int ndim = x->ndim;
      int64_t* shape = x->shape;
      if (axis < 0) axis += ndim;
      ICHECK(axis >= 0 && axis < ndim);

      if (x->dtype.code != kDLFloat || x->dtype.bits != 32 || x->dtype.lanes != 1) {
        LOG(FATAL) << "Not implemented";
      }

      // Set mode and shape descriptor
      if (axis == ndim - 1) {
        int64_t N = 1;
        for (int i = 0; i < ndim - 1; ++i) {
          N *= shape[i];
        }
        uint64_t nBatch = static_cast<uint64_t>(N);
        uint64_t nClass = static_cast<uint64_t>(shape[ndim - 1]);

        vednnError_t e = vednnSoftmaxForward(VEDNN_SOFTMAX_ACCURATE, x->data, y->data, nBatch, nClass);
        ICHECK_EQ(e, VEDNN_SUCCESS) << "vednnSoftmaxForward() failed";
      } else {
        LOG(FATAL) << "Not implemented";
      }
    });

}  // namespace contrib
}  // namespace tvm
