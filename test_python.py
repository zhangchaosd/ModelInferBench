import sys
import time

import numpy as np
import onnxruntime as ort
from openvino.runtime import Core

skip_cpu = True
# onnx_path = "model.onnx"
onnx_path = sys.argv[1]
batch_size = 1
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# get input infos
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
if isinstance(input_shape[0], str):
    input_shape[0] = batch_size
input_type = sess.get_inputs()[0].type
if input_type == "tensor(float)":
    np_type = np.float32
elif input_type == "tensor(uint8)":
    np_type = np.uint8
else:
    print(f"Not impl for other dtype {input_type}")
    exit()


results = []


def test_ONNXRuntime():
    available_providers = ort.get_available_providers()
    if skip_cpu:
        available_providers.remove("CPUExecutionProvider")
    print(f"ONNX Runtime available_providers: {available_providers}")
    for provider in available_providers:
        print(f"ONNX Runtime {provider}...")
        sess = ort.InferenceSession(onnx_path, providers=[provider])
        input = np.random.rand(*input_shape).astype(np_type)
        total_time = 0.0
        epoch = 20
        for i in range(epoch):
            start_time = time.time()
            output = sess.run(["output"], {"input": input})[0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"ONNX Runtime: {provider}. Running time: {elapsed_time:.0f} ms")
            if i >= epoch - 10:
                total_time += elapsed_time
        total_time /= 10
        result = f"ONNX Runtime: {provider}. Average running time in last 10 epochs: {total_time:.0f} ms"
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

        input_tensor = np.random.rand(*input_shape).astype(np_type)
        compiled_model = core.compile_model(model, device_name)

        total_time = 0.0
        epoch = 20
        for i in range(epoch):
            start_time = time.time()
            _ = compiled_model.infer_new_request({0: input_tensor})
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"OpenVINO {device_name}. Running time: {elapsed_time:.0f} ms")
            if i >= epoch - 10:
                total_time += elapsed_time
        total_time /= 10
        result = f"OpenVINO: {device_name}. Average running time in last 10 epochs: {total_time:.0f} ms"
        print(result)
        results.append(result)
        # This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
        # https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html


test_ONNXRuntime()
test_OpenVINO()

print(f"\nResults (input_shape: {input_shape}, dtype:{np_type}):")
list(map(print, results))
