// onnxruntime_windows_cpp_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>

#include <onnxruntime_cxx_api.h>

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    long long elapsedMilliseconds() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

int main()
{
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        std::cout << provider << std::endl;
    }

    const int64_t batch_size = 1;
    std::cout << "batch_size:" << batch_size << std::endl;

    // Initialize ONNX Runtime
    Ort::Env env;

    // Initialize session
    Ort::Session onnx_session(env, L"../../model.onnx", Ort::SessionOptions{ nullptr });

    // Create input tensor objects (This might differ based on your model)
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::unique_ptr<float_t[]> input_image_(new float_t[batch_size * 3 * 224 * 224]);
    std::array<int64_t, 4> input_shape_{ batch_size, 3, 224, 224 };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float_t>(memory_info, input_image_.get(), batch_size * 3 * 224 * 224,
        input_shape_.data(), input_shape_.size());

    // Run model
    std::vector<const char*> input_node_names = { "input" };
    std::vector<const char*> output_node_names = { "output" };

    long long total_time = 0;
    Timer timer;
    for (int i = 0; i < 20; i++) {
        timer.reset();
        auto outputs = onnx_session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        long long elapsed_time = timer.elapsedMilliseconds();
        std::cout << i << " Elapsed time: " << elapsed_time << " ms" << std::endl;
        if (i >= 10) {
            total_time += elapsed_time;
        }
    }
    std::cout << "Running done" << std::endl;
    std::cout << "ONNX Runtime CPU: Average elapsed time: " << total_time / 10 << " ms" << std::endl;
    int a;
    std::cin >> a;

    return 0;
}
