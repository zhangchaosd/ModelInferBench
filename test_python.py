import os
import sys
import time

import numpy as np
import onnxruntime as ort
import openvino as ov

skip_cpu = False
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
    available_providers.remove("TensorrtExecutionProvider")
    if skip_cpu:
        available_providers.remove("CPUExecutionProvider")
    print(f"ONNX Runtime available_providers: {available_providers}")
    for provider in available_providers:
        print(f"ONNX Runtime {provider}...")
        sess = ort.InferenceSession(onnx_path, providers=[provider])
        input = np.random.rand(*input_shape).astype(np_type)
        times = []
        epoch = 20
        for _ in range(epoch):
            start_time = time.time()
            output = sess.run(["output"], {"input": input})[0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"ONNX Runtime: {provider}. Running time: {elapsed_time:.0f} ms")
            times.append(elapsed_time)
        result = f"ONNX Runtime: {provider}. Average running time in last 10 epochs: {np.mean(times[-10:]):.0f} ms"
        print(result)
        results.append(result)


def test_onnxruntime_directml():
    # pip install onnxruntime-directml
    pass


def test_OpenVINO():
    # initialize OpenVINO
    core = ov.Core()

    # print available devices
    devices = core.available_devices
    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    if skip_cpu:
        devices.remove("CPU")

    for device in devices:
        print(f"OpenVINO {device} {core.get_property(device,'FULL_DEVICE_NAME')}...")

        # Construct input
        input_tensor = np.random.rand(*input_shape).astype(np_type)
        c_input_image = np.ascontiguousarray(input_tensor, dtype=np_type)
        input_tensor = ov.Tensor(c_input_image, shared_memory=True)

        config = {"PERFORMANCE_HINT": "LATENCY"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = os.cpu_count()
        ov_model = ov.convert_model(onnx_path)
        compiled_model = core.compile_model(ov_model, device, config=config)

        times = []
        epoch = 20
        for _ in range(epoch):
            start_time = time.time()
            result = compiled_model(input_tensor)[compiled_model.output(0)][0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"OpenVINO {device}. Running time: {elapsed_time:.0f} ms")
            times.append(elapsed_time)
        result = f"OpenVINO: {device}. Average running time in last 10 epochs: {np.mean(times[-10:]):.0f} ms"
        print(result)
        results.append(result)
        del compiled_model


test_ONNXRuntime()
test_OpenVINO()

print(f"\nResults (input_shape: {input_shape}, dtype:{np_type}):")
list(map(print, results))
