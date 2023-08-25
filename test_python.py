import torch
import onnx
import torchvision
import time
import onnxruntime as ort
import numpy as np

onnx_path = "model.onnx"
batch_size = 4

def export_onnx(model):
    model.eval().to("cpu")
    input = torch.randn(batch_size,3,224,224).to("cpu")
    torch.onnx.export(model,
        input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
            }
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

def test(model, device):
    model.eval().to(device)
    input = torch.randn(batch_size,3,224,224).to(device)

    total_time = 0.
    epoch = 20
    for i in range(epoch):
        start_time = time.time()
        output = model(input)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Running time: {elapsed_time:.3f} ms")
        if i >= epoch - 10:
            total_time += elapsed_time
    total_time /= 10
    print(f"Device: {device}. Average running time in last 10 epochs: {total_time:.3f} ms")

def test_onnxruntime(provider):
    sess = ort.InferenceSession(onnx_path, providers=[provider])
    input = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    total_time = 0.
    epoch = 20
    for i in range(epoch):
        start_time = time.time()
        output = sess.run(["output"], {"input":input})[0]
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"Running time: {elapsed_time:.3f} ms")
        if i >= epoch - 10:
            total_time += elapsed_time
    total_time /= 10
    print(f"Onnxruntime: {provider}. Average running time in last 10 epochs: {total_time:.3f} ms")


# model = torchvision.models.efficientnet_v2_l()
model = torchvision.models.efficientnet_b4()
export_onnx(model)

devices = ["cpu", "cuda"]
for device in devices:
    test(model, device)

providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
for provider in providers:
    test_onnxruntime(provider)