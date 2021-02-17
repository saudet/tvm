import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import cc
from tvm.contrib.debugger import debug_runtime
import tvm.contrib.graph_runtime as runtime

def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret


parser = argparse.ArgumentParser(description="Optimize BERT-base model from GluonNLP")
parser.add_argument("-b", "--batch", type=int, default=1,
                    help="Batch size (default: 1)")
parser.add_argument("-l", "--length", type=int, default=128,
                    help="Sequence length (default: 128)")
args = parser.parse_args()
batch = args.batch
seq_length = args.length


# Instantiate a BERT classifier using GluonNLP
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
mx_ctx = mx.cpu()
bert, _ = nlp.model.get_model(
    name=model_name,
    ctx=mx_ctx,
    dataset_name=dataset,
    pretrained=False,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)
model = nlp.model.BERTClassifier(bert, dropout=0.1, num_classes=2)
model.initialize(ctx=mx_ctx)
model.hybridize(static_alloc=True)

# Prepare input data
dtype = "float32"
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length] * batch).astype(dtype)

# Convert to MXNet NDArray and run the MXNet model
inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)
token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)
valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)
mx_out = model(inputs_nd, token_types_nd, valid_length_nd)
mx_out.wait_to_read()

# Benchmark the MXNet latency
res = timer(lambda: model(inputs_nd, token_types_nd, valid_length_nd).wait_to_read(),
            repeat=3,
            dryrun=5,
            min_repeat_ms=1000)
print(f"MXNet latency for batch {batch} and seq length {seq_length}: {np.mean(res):.2f} ms")


######################################
# Optimize the BERT model using TVM
######################################

# First, Convert the MXNet model into TVM Relay format
shape_dict = {
    'data0': (batch, seq_length),
    'data1': (batch, seq_length),
    'data2': (batch,)
}
mod, params = relay.frontend.from_mxnet(model, shape_dict)

# Compile the imported model
#target = "llvm -mcpu=skylake-avx512 -libs=cblas"
#target = "llvm -mcpu=skylake-avx512 -libs=mkl"
target = "llvm"
with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    graph, lib, cparams = relay.build(mod, target, params=params)

# Create the executor and set the parameters and inputs
ctx = tvm.cpu()
rt = runtime.create(graph, lib, ctx)
# or to obtain an execution profile:
#rt = debug_runtime.create(graph, lib, ctx)
rt.set_input(**cparams)
rt.set_input(data0=inputs, data1=token_types, data2=valid_length)

# Run the executor and validate the correctness
rt.run()
out = rt.get_output(0)
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

# Benchmark the TVM latency
ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")


# Export for VE target...
# ...with VPU:
#with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
#    compiled_lib = relay.build(mod, "llvm -mtriple=ve-linux -mattr=+vpu -libs=cblas", params=params)
#
# ...without VPU:
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}, required_pass=["FastMath"]):
    compiled_lib = relay.build(mod, "llvm -mtriple=ve-linux -mattr=-vpu -libs=cblas", params=params)
compiled_lib.export_library("lib/libbertve.so", cc.cross_compiler("/opt/nec/ve/bin/nc++"))

# Create the executor and set the parameters and inputs
ctx = tvm.context("ve", 0)
set_device = tvm.get_global_func("__tvm_set_device")
set_device(ctx.device_type, ctx.device_id)

# The normal dynamic loading method for deployment
tvm.runtime.load_module("lib/libtvm_runtime_pack.so", "vepreload")
tvm.runtime.load_module("lib/libbertve.so", "vepreload")
tvm.runtime.load_module("lib/libtvm_runtime_pack.so", "ve")
loaded_lib = tvm.runtime.load_module("lib/libbertve.so", "ve")
gmod = runtime.GraphModule(loaded_lib["default"](ctx))
# or to obtain an execution profile:
#gmod = debug_runtime.GraphModuleDebug(loaded_lib["debug_create"]("default", ctx), [ctx], compiled_lib.get_json(), '.')

inputs_ve = tvm.nd.array(inputs, ctx)
token_types_ve = tvm.nd.array(token_types, ctx)
valid_length_ve = tvm.nd.array(valid_length, ctx)
gmod.set_input(data0=inputs_ve, data1=token_types_ve, data2=valid_length_ve)

# Run the executor and validate the correctness
gmod.run()
out = gmod.get_output(0)
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

# Benchmark the TVM latency on VE
ftimer = gmod.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency on VE for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")
