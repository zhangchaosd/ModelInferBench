# ModelInferBench

This tool tests ONNX model inference speed using different deployment methods.

## My Results

**Note**: My PC has two graphics cards: GTX1070Ti and Intel A770. My display cable is connected to the A770. In DirectML, device 0 corresponds to A770 and device 1 corresponds to GTX1070Ti.

### System Configurations
#### My PC:

- CPU: i9-13900

- Memory: 32GB DDR4 3000 MHz

- GPUs: GTX1070ti + A770 16G

- OS: Windows 11 Pro 22H2 22621.2215

- GPU Driver version: 536.99 & 31.0.101.4669

- Python Version: 3.11.4

#### My Mac:

- MacBook Pro 16-inch, 2021, A2485

- CPU: Apple M1 Pro

- OS: Ventura 13.5 (22G74)

### Test Parameters

- **Test Model**: `torchvision.models.efficientnet_b4`

- **Input Size**: `batch_size, 3, 224, 224`

- **Inference Runs**: 20 times (the average of the last 10 runs is taken)

- **Unit**: ms

| PC/batch_size | 1 | 4 | 128|
|:------|:----:|:------:|:-:|
| Python PyTorch cpu | 172 ms | 514 ms | * |
| Python PyTorch cuda | 11 ms | 23 ms | * |
| Python onnxruntime cpu | 12 ms | 30 ms | * |
| Python onnxruntime cuda | 7 ms | 18 ms | 430 ms |
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

To test inference speed, either export an ONNX file using the provided Python script or use your own ONNX model. Depending on the model, you may also need to update the file path, input shape, input name, or data type in the code.

### Python

Run the following command to test:

```
python test_python.py
```


### Windows/C++
#### Attention

If you only have CUDA 12.2 installed, you might experience crashes when trying to use the GPU. To avoid this, install CUDA 11.8 to ensure all necessary DLLs are available.

#### onnxruntime:

1 Download the ONNX Runtime release from https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1

2 Download either `onnxruntime-win-x64-1.15.1.zip` or `onnxruntime-win-x64-gpu-1.15.1.zip`

3 Unzip the downloaded file and place its contents in either `windows_sln\onnxruntime_windows_cpp_cpu` or `windows_sln\onnxruntime_windows_cpp_gpu`

3 Depending  on the version you downloaded, you may need to update the following project settings: 
- Properties -> C/C++ -> General -> Additional Include Directories
- Properties -> Linker -> Input -> Additional Dependencies
- Build Events -> Post-Build Event -> Command Line

#### OpenVINO:

1 Download OpenVINO from https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html and unzip it.

2 Create a new folder named `openvino` in `windows_sln\openvino_windows_cpp\`.

3 Copy `openvino_toolkit\runtime\lib` and `openvino_toolkit\runtime\include` to `windows_sln\openvino_windows_cpp\openvino`

4 Build project.

5 Copy all `.dll` files from `openvino_toolkit\runtime\bin\intel64\Release\` and `openvino_toolkit\runtime\3rdparty\tbb\bin\` to the output directory `windows_sln\x64\Release\`.

### Windows/C#

Install one of the following NuGet Packages `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Gpu`.

After adding the ONNX file to the project, change its properties to "Content".
