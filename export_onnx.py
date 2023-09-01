import time

import onnx
import torch
import torchvision

skip_cpu = True
onnx_path = "model.onnx"
batch_size = 1
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


def test_Pytorch(model):
    devices = [] if skip_cpu else ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    # TODO ROCm
    for device in devices:
        print(f"PyTorch {device}...")
        model.eval().to(device)
        input = torch.randn(*input_size).to(device)

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


export_onnx(model)

test_Pytorch(model)
print(f"\nResults (batch_size: {batch_size}):")
list(map(print, results))
