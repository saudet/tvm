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
 * \file ve_common.h
 * \brief Common utilities for VE
 */
#ifndef TVM_RUNTIME_VE_VE_COMMON_H_
#define TVM_RUNTIME_VE_VE_COMMON_H_

#include <ve_offload.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>

#include <string>

#include "../library_module.h"
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of VEs supported */
static constexpr const int kMaxNumVEs = 32;

/*! \brief Thread local workspace */
class VEThreadEntry {
 public:
  /*! \brief The VE process */
  veo_proc_handle* proc{nullptr};
  /*! \brief The VE thread */
  veo_thr_ctxt* thr{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  VEThreadEntry();
  // get the threadlocal workspace
  static VEThreadEntry* ThreadLocal();
};

PackedFunc VEWrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& sptr_to_self);

extern uint64_t VEmemcpy;
extern uint64_t VETVMFuncCall;
extern uint64_t VETVMAPISetLastError;
extern uint64_t VETVMBackendGetFuncFromEnv;
extern uint64_t VETVMBackendAllocWorkspace;
extern uint64_t VETVMBackendFreeWorkspace;
extern uint64_t VETVMBackendParallelLaunch;
extern uint64_t VETVMBackendParallelBarrier;

void VEInitContextFunctions(std::function<void*(const char*)> fgetsymbol);

Module VECreateModuleFromLibrary(ObjectPtr<Library> lib);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VE_VE_COMMON_H_
