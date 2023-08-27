import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

onnx_path = "model.onnx"
batch_size = 1
input_size = (batch_size, 3, 224, 224)
np_type = np.float32
torch_type = torch.float32

# model = torchvision.models.efficientnet_v2_l()
model = torchvision.models.efficientnet_b4()


def export_onnx(model):
    model.eval().to("cpu")
    input = torch.randn(*input_size).to("cpu")
    torch.onnx.export(
        model,
        input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


results = []


def test(model, device):
    model.eval().to(device)
    input = torch.randn(*input_size).to(device).to(torch_type)

    total_time = 0.0
    epoch = 20
    for i in range(epoch):
        start_time = time.time()
        output = model(input)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Running time: {elapsed_time:.0f} ms")
        if i >= epoch - 10:
            total_time += elapsed_time
    total_time /= 10
    result = (
        f"Device: {device}. Average running time in last 10 epochs: {total_time:.0f} ms"
    )
    print(result)
    results.append(result)


def test_onnxruntime(provider):
    sess = ort.InferenceSession(onnx_path, providers=[provider])
    input = np.random.rand(*input_size).astype(np_type)
    total_time = 0.0
    epoch = 20
    for i in range(epoch):
        start_time = time.time()
        output = sess.run(["output"], {"input": input})[0]
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Running time: {elapsed_time:.0f} ms")
        if i >= epoch - 10:
            total_time += elapsed_time
    total_time /= 10
    result = f"Onnxruntime: {provider}. Average running time in last 10 epochs: {total_time:.0f} ms"
    print(result)
    results.append(result)

def test_onnxruntime_directml():
    # pip install onnxruntime-directml
    pass

def test_OpenVINO():
    # python -m pip install openvino
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
    from openvino.runtime import Core, Layout, Type
    print(Core().available_devices)
    device_name = "CPU"  # change here
    # device_name = "GPU"  # change here
    # device_name = "GPU.0"  # A770
    # device_name = "GPU.1"  # 1070Ti

    # Step 1. Initialize OpenVINO Runtime Core
    core = Core()
    # Step 2. Read a model
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(onnx_path)

    if len(model.inputs) != 1:
        print('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        print('Sample supports only single output topologies')
        return -1

    # Step 4. Apply preprocessing
    # ppp = PrePostProcessor(model)

    input_tensor = np.random.rand(*input_size).astype(np_type)

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - reuse precision and shape from already available `input_tensor`
    # - layout of data is 'NHWC'
    # ppp.input().tensor() \
    #     .set_shape(input_size) \
    #     .set_element_type(Type.f32) \
    #     .set_layout(Layout('NCHW'))  # noqa: ECE001, N400

    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    # ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Here we suppose model has 'NCHW' layout for input
    # ppp.input().model().set_layout(Layout('NCHW'))

    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    # ppp.output().tensor().set_element_type(Type.f32)

    # 5) Apply preprocessing modifying the original 'model'
    # model = ppp.build()
    compiled_model = core.compile_model(model, device_name)

    total_time = 0.0
    epoch = 20
    for i in range(epoch):
        start_time = time.time()
        _ = compiled_model.infer_new_request({0: input_tensor})
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"OpenVINI {i} Running time: {elapsed_time:.0f} ms")
        if i >= epoch - 10:
            total_time += elapsed_time
    total_time /= 10
    result = f"OpenVINI: {device_name}. Average running time in last 10 epochs: {total_time:.0f} ms"
    print(result)
    results.append(result)

    # print('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    #https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html
    return 0

export_onnx(model)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
if torch.backends.mps.is_available():
    devices.append("mps")
for device in devices:
    test(model, device)

available_providers = ort.get_available_providers()
for provider in available_providers:
    test_onnxruntime(provider)

test_OpenVINO()

print("Results:")
for result in results:
    print(result)