# ModelInferBench
For testing model inference speed

python test_python.py

## Windows

My PC:
13900
32G DDR4 3000
1070ti + A770 16G

My Mac:
MacBook Pro 16-inch, 2021
Apple M1 Pro
A2485
13.5 (22G74)


efficientnet_b4  1, 3, 224, 224

Infer 20 times, and take the average of the last 10 times.
(ms)

| PC | batch_size 1 | batch_size 4 |
|:------:|:----:|:------:|
| PyTorch cpu | 172 ms | 514 ms |
| PyTorch cuda | 11 ms | 23 ms |
| Python onnxruntime cpu | 12 ms | 30 ms |
| Python onnxruntime cuda | 8 ms | 18 ms |
| C++ onnxruntime cpu | 10 ms | 34 ms |

| MacBook | batch_size 1 | batch_size 4 |
|:------:|:----:|:------:|
| PyTorch cpu | 887 ms | 1207 ms |
| PyTorch mps | 37 ms | 39 ms |
| Python onnxruntime cpu | 59 ms | 208 ms |