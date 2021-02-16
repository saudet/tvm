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
 * \file ve_libary.cc
 * \brief Create library module to load from dynamic shared library for VE target.
 */
#include <ve_offload.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../library_module.h"
#include "ve_common.h"

namespace tvm {
namespace runtime {

// Dynamic shared libary for VE.
class VELibrary final : public Library {
 public:
  ~VELibrary() {
    if (lib_handle_) Unload();
  }
  void Init(const std::string& name) { Load(name); }

  void* GetSymbol(const char* name) final { return GetSymbol_(name); }

  PackedFunc WrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& mptr) {
    return VEWrapPackedFunc(faddr, mptr);
  }

  void InitContextFunctions(std::function<void*(const char*)> fgetsymbol) {
    VEInitContextFunctions(fgetsymbol);
  }

 private:
  // Library handle
  uint64_t lib_handle_{0};
  // load the library
  void Load(const std::string& name) {
    veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
    ICHECK(proc != nullptr) << "Cannot load library without VE process";
    lib_handle_ = veo_load_library(proc, name.c_str());
    ICHECK_NE(lib_handle_, 0) << "Failed to load dynamic shared library " << name;
  }

  void* GetSymbol_(const char* name) {
    veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
    ICHECK(proc != nullptr) << "Cannot get symbol without VE process";
    void* sym = reinterpret_cast<void*>(veo_get_sym(proc, lib_handle_, name));
    return sym;
  }

  void Unload() {
    veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
    ICHECK(proc != nullptr) << "Cannot unload library without VE process";
    ICHECK_EQ(veo_unload_library(proc, reinterpret_cast<uint64_t>(lib_handle_)), 0);
    lib_handle_ = 0;
  }
};

PackedFunc VEWrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& sptr_to_self) {
  return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
    TVMValue ret_value;
    int ret_type_code = kTVMNullptr;

    veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
    veo_thr_ctxt* thr = VEThreadEntry::ThreadLocal()->thr;

    TVMValue values_ve[args.num_args];
    for (int i = 0; i < args.num_args; i++) {
        int type = args.type_codes[0];
        if (type == kTVMDLTensorHandle || type == kTVMNDArrayHandle) {
            DLTensor tensor = *(DLTensor*)args.values[i].v_handle;
            tensor.ctx.device_type = kDLCPU; // from the perspective of the VE code, it's using the CPU runtime
            tensor.ctx.device_id = 0;
            if (tensor.shape != nullptr) {
                uint64_t shape_ve;
                ICHECK_EQ(veo_alloc_mem(proc, &shape_ve, tensor.ndim * sizeof(int64_t)), 0);
                ICHECK_EQ(veo_write_mem(proc, shape_ve, tensor.shape, tensor.ndim * sizeof(int64_t)), 0);
                tensor.shape = (int64_t*)shape_ve;
            }
            if (tensor.strides != nullptr) {
                uint64_t strides_ve;
                ICHECK_EQ(veo_alloc_mem(proc, &strides_ve, tensor.ndim * sizeof(int64_t)), 0);
                ICHECK_EQ(veo_write_mem(proc, strides_ve, tensor.strides, tensor.ndim * sizeof(int64_t)), 0);
                tensor.strides = (int64_t*)strides_ve;
            }
            uint64_t tensor_ve;
            ICHECK_EQ(veo_alloc_mem(proc, &tensor_ve, sizeof(DLTensor)), 0);
            ICHECK_EQ(veo_write_mem(proc, tensor_ve, &tensor, sizeof(DLTensor)), 0);
            values_ve[i].v_handle = (void*)tensor_ve;
        } else {
            values_ve[i] = args.values[i];
        }
    }

    veo_args* argp = veo_args_alloc();
    ICHECK(argp != nullptr) << "veo_args_alloc(): allocation of veo_args failed";

    size_t args_size[] = {args.num_args * sizeof(TVMValue), args.num_args * sizeof(int), 0, sizeof(int), sizeof(int), 0};
    const void* args_vh[] = {values_ve, args.type_codes, (void*)(uint64_t)args.num_args, &ret_value, &ret_type_code, nullptr};
    uint64_t args_ve[6];
    for (int i = 0; i < 6; i++) {
      if (args_size[i] != 0) {
        ICHECK_EQ(veo_alloc_mem(proc, &args_ve[i], args_size[i]), 0);
        ICHECK_EQ(veo_write_mem(proc, args_ve[i], args_vh[i], args_size[i]), 0);
      } else {
        args_ve[i] = (uint64_t)args_vh[i];
      }
      ICHECK_EQ(veo_args_set_u64(argp, i, args_ve[i]), 0);
    }

    long id = veo_call_async(thr, reinterpret_cast<uint64_t>(faddr), argp);
    ICHECK_NE(id, VEO_REQUEST_ID_INVALID) << "veo_call_async(): request failed";

    uint64_t ret = 0;
    ICHECK_EQ(veo_call_wait_result(thr, id, &ret), 0);

    veo_args_free(argp);
    ICHECK_EQ(ret, 0) << TVMGetLastError();
    if (ret_type_code != kTVMNullptr) {
      *rv = TVMRetValue::MoveFromCHost(ret_value, ret_type_code);
    }
  });
}

uint64_t VEmemcpy = 0;
uint64_t VETVMFuncCall = 0;
uint64_t VETVMAPISetLastError = 0;
uint64_t VETVMBackendGetFuncFromEnv = 0;
uint64_t VETVMBackendAllocWorkspace = 0;
uint64_t VETVMBackendFreeWorkspace = 0;
uint64_t VETVMBackendParallelLaunch = 0;
uint64_t VETVMBackendParallelBarrier = 0;

void VEInitContextFunctions(std::function<void*(const char*)> fgetsymbol) {
  veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
  ICHECK(proc != nullptr) << "Cannot init context functions without VE process";
#define TVM_INIT_CONTEXT_FUNC(FuncName)                                       \
  if (auto fp = reinterpret_cast<uint64_t>(fgetsymbol(#FuncName))) {          \
    VE ## FuncName = fp;                                                      \
  }                                                                           \
  if (auto fp = reinterpret_cast<uint64_t>(fgetsymbol("__" #FuncName))) {     \
    ICHECK_EQ(veo_write_mem(proc, fp, &VE ## FuncName, sizeof(uint64_t)), 0); \
  }
  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(memcpy);
  TVM_INIT_CONTEXT_FUNC(TVMFuncCall);
  TVM_INIT_CONTEXT_FUNC(TVMAPISetLastError);
  TVM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
  TVM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);

#undef TVM_INIT_CONTEXT_FUNC
}

Module VECreateModuleFromLibrary(ObjectPtr<Library> lib) {
  veo_proc_handle* proc = VEThreadEntry::ThreadLocal()->proc;
  ICHECK(proc != nullptr) << "Cannot create module from library without VE process";
  lib->InitContextFunctions([lib](const char* fname) { return lib->GetSymbol(fname); });
  // Load the imported modules
  uint64_t dev_mblob = reinterpret_cast<uint64_t>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));
  Module root_mod;
  if (dev_mblob != 0) {
    uint64_t nbytes = 0;
    ICHECK_EQ(veo_read_mem(proc, &nbytes, dev_mblob, sizeof(nbytes)), 0);

    char *mblob = (char*)malloc(sizeof(nbytes) + nbytes);
    ICHECK(mblob != nullptr) << "Could not allocate memory for module blob";
    ICHECK_EQ(veo_read_mem(proc, mblob, dev_mblob, sizeof(nbytes) + nbytes), 0);
    root_mod = ProcessModuleBlob(mblob, lib);
    free(mblob);
  } else {
    // Only have one single DSO Module
    root_mod = ProcessModuleBlob(nullptr, lib);
  }

  // allow lookup of symbol from root (so all symbols are visible).
// but device code can't access host memory anyway, so we shouldn't set it
//  if (auto ctx_addr = reinterpret_cast<uint64_t>(lib->GetSymbol(runtime::symbol::tvm_module_ctx))) {
//    ICHECK_EQ(veo_write_mem(proc, ctx_addr, root_mod.operator->(), sizeof(uint64_t)), 0);
//  }

  return root_mod;
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ve").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = make_object<VELibrary>();
  n->Init(args[0]);
  *rv = VECreateModuleFromLibrary(n);
});

TVM_REGISTER_GLOBAL("runtime.module.loadfile_vepreload").set_body([](TVMArgs args, TVMRetValue* rv) {
  const std::string& name = args[0].operator std::string();
  veo_proc_handle* proc = tvm::runtime::VEThreadEntry::ThreadLocal()->proc;
  ICHECK(proc != nullptr) << "Cannot preload library without VE process";
  uint64_t lib_handle = veo_load_library(proc, name.c_str());
  ICHECK_NE(lib_handle, 0) << "Failed to preload dynamic shared library " << name;
});
}  // namespace runtime
}  // namespace tvm
