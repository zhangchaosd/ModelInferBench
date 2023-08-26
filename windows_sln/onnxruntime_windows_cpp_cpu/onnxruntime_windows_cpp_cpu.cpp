// onnxruntime_windows_cpp_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>

#include <onnxruntime_cxx_api.h>

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    // 重置计时器
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // 获取经过的毫秒数（不保留小数）
    long long elapsedMilliseconds() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

int main()
{
    std::cout << "Hello World!\n";

    // Initialize ONNX Runtime
    Ort::Env env;

    // Initialize session
    Ort::Session onnx_session(env, L"model.onnx", Ort::SessionOptions{ nullptr });

    // Create input tensor objects (This might differ based on your model)

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float_t, 3 * 224 * 224> input_image_{};
    std::array<int64_t, 4> input_shape_{ 1, 3, 224, 224 };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, input_image_.data(), input_image_.size(),
        input_shape_.data(), input_shape_.size());

    // Run model
    std::vector<const char*> input_node_names = { "input" };
    std::vector<const char*> output_node_names = { "output" };

    long long total_time = 0;
    for (int i = 0; i < 20; i++) {
        Timer timer;
        auto outputs = onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        long long elapsed_time = timer.elapsedMilliseconds();
        std::cout << i << " Elapsed time: " << elapsed_time << " ms" << std::endl;
        if (i >= 10) {
            total_time += elapsed_time;
        }
    }
    printf("Running done");
    std::cout << "Average elapsed time: " << total_time / 10 << " ms" << std::endl;
    int a;
    std::cin >> a;

    return 0;
}
