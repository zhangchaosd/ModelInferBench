import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

onnx_path = "model.onnx"
batch_size = 4
input_size = (batch_size, 3, 224, 224)
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
    input = torch.randn(*input_size).to(device)

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
    input = np.random.rand(*input_size).astype(np.float32)
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

print("Results:")
for result in results:
    print(result)
