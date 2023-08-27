# ModelInferBench
This is a tool for testing onnx model inference speed in different deploy methods.


## My Results

Notice: My PC has two graphics cards: GTX1070Ti and Intel A770. My display cable is on A770. In DirectML, device 0 is A770 and device 1 is GTX1070Ti.

My PC:

- CPU: i9-13900

- MEM: 32G DDR4 3000

- GPUs: GTX1070ti + A770 16G

- System: Windows 11 Pro 22H2 22621.2215

- Python 3.11.4

My Mac:

- MacBook Pro 16-inch, 2021, A2485

- Apple M1 Pro

- Ventura 13.5 (22G74)


Test model: torchvision.models.efficientnet_b4

Input size: batch_size, 3, 224, 224

Infer 20 times, and take the average of the last 10 times.
(ms)

| PC/batch_size | 1 | 4 | 128|
|:------|:----:|:------:|:-:|
| Python PyTorch cpu | 172 ms | 514 ms | * |
| Python PyTorch cuda | 11 ms | 23 ms | * |
| Python onnxruntime cpu | 12 ms | 30 ms | * |
| Python onnxruntime cuda | 8 ms | 18 ms | 430 ms |
| C++ onnxruntime cpu | 10 ms | 34 ms | 3800 ms |
| C++ onnxruntime cuda | 7 ms | 17 ms | 424 ms |
| C# onnxruntime cpu | 170 ms | 473 ms | 3876 ms |
| C# onnxruntime cuda | 7 ms | 17 ms | 427 ms|
| C# DirectML A770 | 9 ms | 19 ms | 485 ms|
| C# DirectML 1070Ti | 12 ms | 31 ms | 812 ms|
| Python OpenVINO CPU | 11 ms | 29 ms | * |  |
| Python OpenVINO A770 | 10 ms | 15 ms | 919 ms |
| Python OpenVINO 1070Ti | 49 ms | * | * |
| C++ OpenVINO CPU | 10 ms | 26 ms | * |
| C++ OpenVINO A770 | 7 ms | 10 ms | 870 ms |

| MacBook/batch_size | 1 | 4 |
|:------|:----:|:------:|
| Python PyTorch cpu | 887 ms | 1207 ms |
| Python PyTorch mps | 37 ms | 39 ms |
| Python onnxruntime cpu | 59 ms | 208 ms |


## Instructions

### Python

```
python test_python.py
```


### Windows/C++
#### Attention

If you only install cuda12.2, you may can't use gpu(crash). You need to install cuda 11.8, so there is some necessary dlls.

#### onnxruntime:

1 Download onnxruntime release from https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1

2 Download `onnxruntime-win-x64-1.15.1.zip` or `onnxruntime-win-x64-gpu-1.15.1.zip`

3 Unzip and put them in `windows_sln\onnxruntime_windows_cpp_cpu` or `windows_sln\onnxruntime_windows_cpp_gpu`

3 Depends on the version you download, you may need to change following project settings: 
- Properties -> C/C++ -> General -> Additional Include Directories
- Properties -> Linker -> Input -> Additional Dependencies
- Build Events -> Post-Build Event -> Command Line

#### OpenVINO:

1 Download from https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html and unzip.

2 Create a new folder named `openvino` in `windows_sln\openvino_windows_cpp\`.

3 Copy `openvino_toolkit\runtime\lib` and `openvino_toolkit\runtime\include` to `windows_sln\openvino_windows_cpp\openvino`

4 Build project.

5 Copy `openvino_toolkit\runtime\bin\intel64\Release\*.dll` and `openvino_toolkit\runtime\3rdparty\tbb\bin\*.dll` to the output path `windows_sln\x64\Release\`.

### Windows/C#

Install one of the following NuGet Package `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Gpu`

After add onnx to project, you have to change the file's property to content.
