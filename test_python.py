import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from openvino.runtime import Core

skip_cpu = True
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


def test_Pytorch(model):
    devices = [] if skip_cpu else ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    for device in devices:
        print(f"OpenVINO {device}...")
        model.eval().to(device)
        input = torch.randn(*input_size).to(device).to(torch_type)

        total_time = 0.0
        epoch = 20
        for i in range(epoch):
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"Pytorch: {device}. Running time: {elapsed_time:.0f} ms")
            if i >= epoch - 10:
                total_time += elapsed_time
        total_time /= 10
        result = f"Pytorch: {device}. Average running time in last 10 epochs: {total_time:.0f} ms"
        print(result)
        results.append(result)


def test_ONNXRuntime():
    available_providers = ort.get_available_providers()
    if skip_cpu:
        available_providers.remove("CPUExecutionProvider")
    print(f"ONNX Runtime available_providers: {available_providers}")
    for provider in available_providers:
        print(f"OpenVINO {provider}...")
        sess = ort.InferenceSession(onnx_path, providers=[provider])
        input = np.random.rand(*input_size).astype(np_type)
        total_time = 0.0
        epoch = 20
        for i in range(epoch):
            start_time = time.time()
            output = sess.run(["output"], {"input": input})[0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"Onnxruntime: {provider}. Running time: {elapsed_time:.0f} ms")
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
    devices = Core().available_devices
    if skip_cpu:
        devices.remove("CPU")
    print(f"OpenVINO available devices: {devices}")
    for device_name in devices:
        print(f"OpenVINO {device_name}...")
        core = Core()
        # Read a model
        # (.xml and .bin files) or (.onnx file)
        model = core.read_model(onnx_path)

        if len(model.inputs) != 1:
            print("Sample supports only single input topologies")
            return

        if len(model.outputs) != 1:
            print("Sample supports only single output topologies")
            return

        input_tensor = np.random.rand(*input_size).astype(np_type)
        compiled_model = core.compile_model(model, device_name)

        total_time = 0.0
        epoch = 20
        for i in range(epoch):
            start_time = time.time()
            _ = compiled_model.infer_new_request({0: input_tensor})
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"OpenVINI {device_name}. Running time: {elapsed_time:.0f} ms")
            if i >= epoch - 10:
                total_time += elapsed_time
        total_time /= 10
        result = f"OpenVINI: {device_name}. Average running time in last 10 epochs: {total_time:.0f} ms"
        print(result)
        results.append(result)
        # This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
        # https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html


export_onnx(model)

test_Pytorch(model)
test_ONNXRuntime()
test_OpenVINO()

print(f"\nResults (batch_size: {batch_size}):")
list(map(print, results))
