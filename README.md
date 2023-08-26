# ModelInferBench
This is a toolset for testing onnx model inference speed in different deploy methods.


## My Results

Notice: My PC has two graphics cards: GTX1070Ti and Intel A770. My display cable is on A770. In DirectML, device 0 is A770 and device 1 is GTX1070Ti.

My PC:

- CPU: 13900

- MEM: 32G DDR4 3000

- GPUs: 1070ti + A770 16G

- System: Windows 11 Pro 22H2 22621.2215

- Python 3.11.4

My Mac:

- MacBook Pro 16-inch, 2021, A2485

- Apple M1 Pro

- 13.5 (22G74)


Test model: efficientnet_b4

Input size: batch_size, 3, 224, 224

Infer 20 times, and take the average of the last 10 times.
(ms)

| PC/batch_size | 1 | 4 | 128| 512 |
|:------:|:----:|:------:|:-:| :-: |
| PyTorch cpu | 172 ms | 514 ms | ms |
| PyTorch cuda | 11 ms | 23 ms | oom |
| Python onnxruntime cpu | 12 ms | 30 ms | ms |
| Python onnxruntime cuda | 8 ms | 18 ms | 430 ms |
| C++ onnxruntime cpu | 10 ms | 34 ms | 3800 ms |
| C++ onnxruntime cuda | 7 ms | 17 ms | 424 ms |
| C# onnxruntime cpu | 170 ms | 473 ms | 3876 ms |
| C# onnxruntime cuda | 7 ms | 17 ms | 427 ms|
| C# DirectML A770 | 9 ms | 19 ms | 485 ms|
| C# DirectML 1070Ti | 12 ms | 31 ms | 812 ms|
| Python OpenVINO CPU | 11 ms | 25 ms | 1184 ms |  |
| Python OpenVINO A770 | 6 ms | 7 ms | 75 ms | 512 ms |
| Python OpenVINO 1070Ti | 49 ms | * | * | * |
| C++ OpenVINO CPU | 10 ms | 26 ms | * | * |
| C++ OpenVINO A770 | 7 ms | 10 ms | 870 ms | * |
| C++ OpenVINO 1070Ti | 49 ms | * | * | * |

| MacBook/batch_size | 1 | 4 |
|:------:|:----:|:------:|
| PyTorch cpu | 887 ms | 1207 ms |
| PyTorch mps | 37 ms | 39 ms |
| Python onnxruntime cpu | 59 ms | 208 ms |

## Attention

If you only install cuda12.2, then you can't use gpu. You have to install cuda 11.8, so there is some necessary dlls.

## Instructions

### Python

python test_python.py


### Windows/C++

1 Download onnxruntime release from https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1

2 Download `onnxruntime-win-x64-1.15.1.zip` or `onnxruntime-win-x64-gpu-1.15.1.zip`

3 Unzip and put them in `windows_sln\onnxruntime_windows_cpp_cpu` or `windows_sln\onnxruntime_windows_cpp_gpu`

3 Depends on the version you download, you may need to change project settings: 
- Properties -> C/C++ -> General -> Additional Include Directories
- Properties -> Linker -> Input -> Additional Dependencies
- Build Events -> Post-Build Event -> Command Line

### Windows/C#

Just install one of the following NuGet Package `Microsoft.ML.OnnxRuntime.DirectML`/`Microsoft.ML.OnnxRuntime`/`Microsoft.ML.OnnxRuntime.Gpu`


