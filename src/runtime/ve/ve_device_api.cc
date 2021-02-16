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
 * \file ve_device_api.cc
 * \brief VE specific API
 */
#include <ve_offload.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include "ve_common.h"

namespace tvm {
namespace runtime {

class VEDeviceAPI final : public DeviceAPI {
 public:
  VEDeviceAPI() : procs_(), thrs_() { }
  void SetDevice(TVMContext ctx) final {
    if (procs_[ctx.device_id] == nullptr) {
      procs_[ctx.device_id] = veo_proc_create(ctx.device_id);
      ICHECK(procs_[ctx.device_id] != nullptr) << "veo_proc_create() failed for device " << ctx.device_id;
      thrs_[ctx.device_id] = static_cast<veo_thr_ctxt*>(CreateStream(ctx));
    }
    VEThreadEntry::ThreadLocal()->proc = procs_[ctx.device_id];
    VEThreadEntry::ThreadLocal()->thr = thrs_[ctx.device_id];
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        SetDevice(ctx);
        value = procs_[ctx.device_id] != nullptr;
        break;
      case kApiVersion: {
        *rv = VEO_API_VERSION;
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    void* ret;
    SetDevice(ctx);
    ICHECK_EQ(veo_alloc_mem(procs_[ctx.device_id], (uint64_t*)&ret, nbytes), 0);
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    SetDevice(ctx);
    ICHECK_EQ(veo_free_mem(procs_[ctx.device_id], (uint64_t)ptr), 0);
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    // In case there is a copy from host mem to host mem */
    if (ctx_to.device_type == kDLCPU && ctx_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (ctx_from.device_type == kDLVE && ctx_to.device_type == kDLVE) {
      SetDevice(ctx_from);
      if (ctx_from.device_id == ctx_to.device_id) {
        veo_args* argp = veo_args_alloc();
        ICHECK(argp != nullptr) << "veo_args_alloc(): allocation of veo_args failed";

        ICHECK_EQ(veo_args_set_u64(argp, 0, reinterpret_cast<uint64_t>(to)), 0);
        ICHECK_EQ(veo_args_set_u64(argp, 1, reinterpret_cast<uint64_t>(from)), 0);
        ICHECK_EQ(veo_args_set_u64(argp, 2, size), 0);

        long id = veo_call_async(thrs_[ctx_from.device_id], reinterpret_cast<uint64_t>(VEmemcpy), argp);
        ICHECK_NE(id, VEO_REQUEST_ID_INVALID) << "veo_call_async(): request failed for memcpy()";

        uint64_t ret = 0;
        ICHECK_EQ(veo_call_wait_result(thrs_[ctx_from.device_id], id, &ret), 0);

        veo_args_free(argp);
      } else {
        LOG(FATAL) << "Device does not support copy between VE";
      }
    } else if (ctx_from.device_type == kDLVE && ctx_to.device_type == kDLCPU) {
      SetDevice(ctx_from);
      ICHECK_EQ(veo_read_mem(procs_[ctx_from.device_id], to, reinterpret_cast<uint64_t>(from), size), 0);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLVE) {
      SetDevice(ctx_to);
      ICHECK_EQ(veo_write_mem(procs_[ctx_to.device_id], reinterpret_cast<uint64_t>(to), from, size), 0);
    } else {
      LOG(FATAL) << "expect copy from/to VE or between VE";
    }
  }

  TVMStreamHandle CreateStream(TVMContext ctx) {
    SetDevice(ctx);
    veo_thr_ctxt* thr = veo_context_open(procs_[ctx.device_id]);
    ICHECK(thr != nullptr) << "veo_context_open() failed";
    return static_cast<TVMStreamHandle>(thr);
  }

  void FreeStream(TVMContext ctx, TVMStreamHandle stream) {
    SetDevice(ctx);
    veo_thr_ctxt* thr = static_cast<veo_thr_ctxt*>(stream);
    ICHECK_EQ(veo_context_close(thr), 0);
  }

  void SyncStreamFromTo(TVMContext ctx, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    LOG(FATAL) << "Not implemented";
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final { }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    VEThreadEntry::ThreadLocal()->thr = static_cast<veo_thr_ctxt*>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    return VEThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    VEThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static VEDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new VEDeviceAPI();
    return inst;
  }

 private:
  // The VE processes, one per device
  veo_proc_handle* procs_[kMaxNumVEs];
  // The VE threads, one per stream
  veo_thr_ctxt* thrs_[kMaxNumVEs];
};

typedef dmlc::ThreadLocalStore<VEThreadEntry> VEThreadStore;

VEThreadEntry::VEThreadEntry() : pool((DLDeviceType)kDLVE, VEDeviceAPI::Global()) {}

VEThreadEntry* VEThreadEntry::ThreadLocal() { return VEThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.ve").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = VEDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace tvm
