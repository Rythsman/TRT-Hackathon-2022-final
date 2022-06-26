
import torch
import multiprocessing
import sys
import os

sys.path.insert(0, os.path.abspath('../ml-cvnets'))

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from cvnets import get_model

from options.opts import get_eval_arguments
from utils import logger
from utils.common_utils import device_setup

import onnx
import os
import numpy as np

from cuda import cudart
import tensorrt as trt
import torch.nn.functional as F
import onnx_graphsurgeon as gs
from time import time_ns
from functools import partial

from glob import glob
import ctypes

trt_logger = trt.Logger(trt.Logger.VERBOSE)
pt_filePath = '../ml-cvnets/pretrained_models/mobilevit_s.pt'
output_onnx = "mobilevit_s.onnx"
epsilon = 1.0e-2
np.random.seed(97)
eps = 1e-6
use_fp16 = False

def GiB(val):
    return val * 1 << 30

def load_weights(weight_path:str):
    suffix = weight_path.split('.')[-1]
    if suffix != 'npz':
        print("Unsupport weight file: Unrecognized format %s " % suffix)
        exit(-1)
    return np.load(weight_path)

#替换
def editOnnxWithFt(onnx_file):
    model = onnx.load(onnx_file)
    model_gs = gs.import_onnx(model)

    input2nodelist = dict()
    output2nodelist = dict()

    def update_dict():
        nonlocal input2nodelist
        nonlocal output2nodelist
        model_gs.cleanup().toposort()
        input2nodelist.clear()
        output2nodelist.clear()
        for node in model_gs.nodes:
            for input_var in node.inputs:
                d = input2nodelist.setdefault(input_var.name, list())
                d.append(node)

        
        for node in model_gs.nodes:
            for output_var in node.outputs:
                d = output2nodelist.setdefault(output_var.name, list())
                d.append(node)

    update_dict()

    for node in model_gs.nodes:
        if node.name == "Reshape_51":
            node_Reshape_51 = node
        if node.name == "Reshape_168":
            node_Reshape_168 = node
        if node.name == "Reshape_200":
            node_Reshape_200 = node
        if node.name == "Reshape_421":
            node_Reshape_421 = node
        if node.name == "Reshape_453":
            node_Reshape_453 = node
        if node.name == "Reshape_622":
            node_Reshape_622 = node
        
    
    nb = 0
    arr_type = np.float16 if use_fp16 else np.float32

    # mobilevit_block1 
    '''
    Reshape_51: input.184
    Add_166: patches
    '''
    typeid = 1 if use_fp16 else 0
    node_transform_block_out_0 = gs.Variable(f"transform_block{nb}_out")
    node_transform_block_0 = gs.Node("CustomMobileVisionTransformerPlugin", f"node_transform_block{nb}", \
            inputs=[node_Reshape_51.outputs[0]], outputs=[node_transform_block_out_0])
    node_transform_block_0.attrs['max_batch'] = np.int32([128])
    node_transform_block_0.attrs['seq_len'] = np.int32([256])
    node_transform_block_0.attrs['embed_dim'] = np.int32([144])
    node_transform_block_0.attrs['num_heads'] = np.int32([4])
    node_transform_block_0.attrs['inter_size'] = np.int32([288])
    node_transform_block_0.attrs['layer_num'] = np.int32([2])
    node_transform_block_0.attrs['type_id'] = np.int32([typeid]) # 0 代表fp32 1 代表fp16
    node_transform_block_0.attrs['npz_path'] = "/root/trt2022_src/mobilenet/tmp/test01_%s.npz" % typeid
    node_transform_block_0.attrs['npz_path_len'] = np.int32([len(node_transform_block_0.attrs['npz_path'])])

    model_gs.nodes.append(node_transform_block_0)
    node_Reshape_168.inputs[0] = node_transform_block_out_0
    for nd in input2nodelist[node_Reshape_51.outputs[0].name]:
        nd.inputs.clear()
    nb += 1 

    # mobilevit_block2
    '''
    Reshape_200: input.300
    Add_419: patches.3
    '''
    node_transform_block_out_1 = gs.Variable(f"transform_block{nb}_out")
    node_transform_block_1 = gs.Node("CustomMobileVisionTransformerPlugin", f"node_transform_block{nb}", \
            inputs=[node_Reshape_200.outputs[0]], outputs=[node_transform_block_out_1])
    node_transform_block_1.attrs['max_batch'] = np.int32([128])
    node_transform_block_1.attrs['seq_len'] = np.int32([64])
    node_transform_block_1.attrs['embed_dim'] = np.int32([192])
    node_transform_block_1.attrs['num_heads'] = np.int32([4])
    node_transform_block_1.attrs['inter_size'] = np.int32([384])
    node_transform_block_1.attrs['layer_num'] = np.int32([4])
    node_transform_block_0.attrs['type_id'] = np.int32([typeid]) # 0 代表fp32 1 代表fp16
    node_transform_block_1.attrs['npz_path'] = "/root/trt2022_src/mobilenet/tmp/test02_%s.npz" % typeid
    node_transform_block_1.attrs['npz_path_len'] = np.int32([len(node_transform_block_1.attrs['npz_path'])])

    model_gs.nodes.append(node_transform_block_1)
    node_Reshape_421.inputs[0] = node_transform_block_out_1
    for nd in input2nodelist[node_Reshape_200.outputs[0].name]:
        nd.inputs.clear()
    nb += 1 

    # mobilevit_block3
    '''
    Reshape_453:  input.456
    Add_419: patches.7
    '''
    node_transform_block_out_2 = gs.Variable(f"transform_block{nb}_out")
    node_transform_block_2 = gs.Node("CustomMobileVisionTransformerPlugin", f"node_transform_block{nb}", \
            inputs=[node_Reshape_453.outputs[0]], outputs=[node_transform_block_out_2])
    node_transform_block_2.attrs['max_batch'] = np.int32([128])
    node_transform_block_2.attrs['seq_len'] = np.int32([16])
    node_transform_block_2.attrs['embed_dim'] = np.int32([240])
    node_transform_block_2.attrs['num_heads'] = np.int32([4])
    node_transform_block_2.attrs['inter_size'] = np.int32([480])
    node_transform_block_2.attrs['layer_num'] = np.int32([3])
    node_transform_block_2.attrs['type_id'] = np.int32([typeid]) # 0 代表fp32 1 代表fp16
    node_transform_block_2.attrs['npz_path'] = "/root/trt2022_src/mobilenet/tmp/test03_%s.npz" % typeid
    node_transform_block_2.attrs['npz_path_len'] = np.int32([len(node_transform_block_2.attrs['npz_path'])])

    model_gs.nodes.append(node_transform_block_2)
    node_Reshape_622.inputs[0] = node_transform_block_out_2
    for nd in input2nodelist[node_Reshape_453.outputs[0].name]:
        nd.inputs.clear()
    nb += 1 

    model_gs.cleanup().toposort()
    model = gs.export_onnx(model_gs)
    basename_wo_ext = os.path.splitext(os.path.basename(onnx_file))[0]
    new_onnx_file = os.path.join(os.path.dirname(onnx_file), basename_wo_ext + "_transformerBlock.onnx")
    onnx.save(model, new_onnx_file)
    return new_onnx_file

def edit_onnx(onnx_file):

    model = onnx.load(onnx_file)
    model_gs = gs.import_onnx(model)

    input2nodelist = dict()
    output2nodelist = dict()

    def update_dict():
        nonlocal input2nodelist
        nonlocal output2nodelist
        model_gs.cleanup().toposort()
        input2nodelist.clear()
        output2nodelist.clear()
        for node in model_gs.nodes:
            for input_var in node.inputs:
                d = input2nodelist.setdefault(input_var.name, list())
                d.append(node)

        
        for node in model_gs.nodes:
            for output_var in node.outputs:
                d = output2nodelist.setdefault(output_var.name, list())
                d.append(node)

    update_dict()

    # 替换layernorm
    for node in model_gs.nodes:
        # 寻找 Add
        break
        if node.op == "Reshape" or node.op == "Add":
            
            if len(node.outputs) == 0:
                continue
            
            output_nodes = input2nodelist[node.outputs[0].name]
            seen_reducemean = 0
            seen_sub = 0
            sub_node = None
            for output_node in output_nodes:
                if output_node.op == "Sub":
                    sub_node = output_node
                    seen_sub += 1
                elif output_node.op == "ReduceMean":
                    seen_reducemean += 1

            if not (1 == seen_reducemean and 1 == seen_sub):
                continue
            
            output_nodes = input2nodelist[sub_node.outputs[0].name]

            if len(output_nodes) != 2:
                continue
            input_var = node.outputs[0]

            pow_node, div_node = output_nodes
            if pow_node.op != 'Pow':
                pow_node, div_node = div_node, pow_node
            def get_next_node(node):
                return input2nodelist[node.outputs[0].name][0]
            add_node = get_next_node(get_next_node(pow_node))

            epsilon = add_node.inputs[1].inputs[0].attrs['value'].values
            mul_node = get_next_node(div_node)

            weight = mul_node.inputs[1].values
            add_node = get_next_node(mul_node)
            bias = add_node.inputs[1].values

            norm_name = "_".join(add_node.inputs[1].name.split(".")[:-1])
            weight_const = gs.Constant('weight_'+norm_name, values=weight)
            bias_const = gs.Constant('bias_'+norm_name, values=bias)
            outputs_vars = add_node.outputs[0:]
            add_node.outputs.clear()
            layernorm_node = gs.Node("lnplugin", name = 'layernorm-'+norm_name, inputs=[input_var, weight_const, bias_const], outputs=outputs_vars)

            layernorm_node.attrs["num_groups"] = 1
            layernorm_node.attrs["eps"] = float(epsilon)

            model_gs.nodes.append(layernorm_node)

    
    update_dict()
    
    # 替换 attention
    for node in model_gs.nodes:
        break
        # 寻找 Add
        if node.op == "Transpose":
            transpose_node = node
            next_nodes = input2nodelist[node.outputs[0].name]
            
            if len(next_nodes) != 3:
                continue
            is_continue = True
            v_multi_node = None
            for next_node in next_nodes:
                if next_node.op != "Gather":
                    is_continue = False
                    break
                if next_node.o().op == "MatMul":
                    v_multi_node = next_node.o()
            if not is_continue:
                break
            N, H = transpose_node.i(0).inputs[1].inputs[0].attrs['value'].values.tolist()[-2:]
            # print(N, H)
            transpose_node.attrs['perm'] = [1,0,3,2,4]
            output_transpose_node = v_multi_node.o()
            output_transpose_node.attrs['perm'] = [1,0,2,3]
            v_multi_node.outputs.clear()

            transpose_output_var = transpose_node.outputs[0]
            qkv_outputs_var = gs.Variable(name = transpose_output_var.name + "_transposed")
            qkvToContextNode = gs.Node(op="mobilebitQKVToContextPlugin", name = transpose_output_var.name + "_transposed_node", inputs=[transpose_output_var], outputs=[qkv_outputs_var])
            qkvToContextNode.attrs['type_id'] = 0
            qkvToContextNode.attrs['hidden_size'] = [H]
            qkvToContextNode.attrs['num_heads'] = [N]
            qkvToContextNode.attrs['has_mask'] = [0]
            output_transpose_node.inputs[0] = qkv_outputs_var
            model_gs.nodes.append(qkvToContextNode)
    
    update_dict()


    # 替换swish
    swish_cnt = 0
    for node in model_gs.nodes:
        if node.op == "Conv":
            
            next_nodes = input2nodelist[node.outputs[0].name]
            if len(next_nodes) != 2:
                continue
            
            node1, node2 = next_nodes
            if not ((node1.op == 'Mul' and node2.op == "Sigmoid") or (node1.op == "Sigmoid" and node2.op == "Mul")):
                continue
            
            
            prev_mul_node = node1
            if prev_mul_node.op != "Mul":
                prev_mul_node = node2
            next_node = prev_mul_node.o()
            
            swish_input_var = gs.Variable(name="swish_input_%d" % (swish_cnt + 1), dtype=node.outputs[0].dtype, shape=node.outputs[0].shape)
            swish_output_var = gs.Variable(name="swish_output_%d" % (swish_cnt + 1), dtype=node.outputs[0].dtype, shape=node.outputs[0].shape)
            node.outputs[0] = swish_input_var
            for i, item in enumerate(next_node.inputs):
                if item.name == prev_mul_node.outputs[0].name:
                    next_node.inputs[i] = swish_output_var
                    break

            swish_node = gs.Node(op="Swish", name="Swish_%d" % (swish_cnt + 1), inputs=[swish_input_var], outputs=[swish_output_var])

            swish_cnt += 1
            for next_node in next_nodes:
                next_node.outputs.clear()
                next_node.inputs.clear()
            model_gs.nodes.append(swish_node)
    
    update_dict()

    model = gs.export_onnx(model_gs)
    basename_wo_ext = os.path.splitext(os.path.basename(onnx_file))[0]
    new_onnx_file = os.path.join(os.path.dirname(onnx_file), basename_wo_ext + "_swish.onnx")
    onnx.save(model, new_onnx_file)
    return new_onnx_file

def buildTrtEngine():
    max_batch_size = 128
    trtFile = "./mobilevit_s.plan"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        print("Succeeded loading engine!")
    else:
        with trt.Builder(trt_logger) as builder, builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ) as network, trt.OnnxParser(network, trt_logger) as parser:
            
            builder.max_batch_size = max_batch_size
            profile = builder.create_optimization_profile()
            profile.set_shape('input', min=(1, 3, 256, 256), opt=(max_batch_size, 3, 256, 256), max=(max_batch_size, 3, 256, 256))

            config = builder.create_builder_config()
            config.max_workspace_size = GiB(18) # 8G
            with open(output_onnx, 'rb') as model:
                parser.parse(model.read())
                for i in range(parser.num_errors):
                    print(parser.get_error(i))

            config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
            if builder.platform_has_fast_fp16 and use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                network.get_output(0).dtype = trt.float16

            # config.flags = 1 << int(trt.BuilderFlag.FP16)
            config.add_optimization_profile(profile)
            engineString = builder.build_serialized_network(network, config)
            if engineString == None:
                print("Failed building engine!")
                return
            print("Succeeded building engine!")
            with open(trtFile, 'wb') as f:
                f.write(engineString)
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engineString)
    return engine

def inferUseTrt(data):
    engine = buildTrtEngine()
    context = engine.create_execution_context()
    context.set_binding_shape(0, tuple(data.shape))
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    print(f"nInput: {nInput}")
    nOutput = engine.num_bindings - nInput

    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

    bufferH = []
    bufferH.append(data)

    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    
    output = bufferH[-1]

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)

    return output, engine


def trt_model(torch_data, engine):

    data = torch_data.cpu().numpy()
    if hasattr(trt_model, 'context'):
        context = getattr(trt_model, 'context')
    else:
        context = engine.create_execution_context()
        setattr(trt_model, 'context', context)

    context.set_binding_shape(0, tuple(data.shape))
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])

    nOutput = engine.num_bindings - nInput

    bufferH = []
    bufferH.append(data)

    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    
    output = bufferH[-1]

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)

    return torch.as_tensor(output).to(torch_data.device)

def main(opts, **kwargs):
    global output_onnx

    # set-up the model
    device = torch.device('cpu')
    model = get_model(opts)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(pt_filePath, map_location='cpu'))

    # set-up data loaders
    model.to(device)
    
    # Generate input tensor with random values
    input_tensor = torch.randn(1, 3, 256, 256).clip(-3, 3)
    res_pytorch = model(input_tensor)
    print("pytorch input shape: ", input_tensor.shape, input_tensor.dtype)
    print("pytorch output shape: ", res_pytorch.shape, res_pytorch.dtype)

    # Export torch model to ONNX
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
        opset_version = 13,
        do_constant_folding = True,
        input_names = ["input"],
        output_names = ["output"],
        dynamic_axes = {
            "input": {
                0: "b",
            },
            "output": {
                0: "b",
            },
        },
        verbose=False
    )
    
    print("Exporting ONNX model {} Success!".format(output_onnx))

    # output_onnx = edit_onnx(output_onnx)

    print('output_onnx', output_onnx)
    # 使用ft plugin
    # output_onnx = editOnnxWithFt(output_onnx)

    data = input_tensor.numpy()
    outputs, engine = inferUseTrt(data)

    print("trt input shape: ", data.shape, data.dtype)
    print("trt output shape: ", outputs.shape, outputs.dtype)

    diff0 = np.max( np.abs(res_pytorch.cpu().detach().numpy() - outputs) )
    diff1 = np.max( np.abs(res_pytorch.cpu().detach().numpy() - outputs) / (eps + np.abs(outputs)) )

    print("check: ", diff0, diff1)



def main_worker(**kwargs):
    opts = get_eval_arguments()
    opts = device_setup(opts)
    num_gpus = 1
    n_cpus = os.cpu_count()
    setattr(opts, "dataset.workers", n_cpus)
    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)
    main(opts=opts, **kwargs)


if __name__ == "__main__":
    main_worker()
