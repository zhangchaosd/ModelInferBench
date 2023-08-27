#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <random>
#include <chrono>

#include "openvino/openvino.hpp"


using namespace ov::preprocess;

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

int main(int argc, char* argv[]) {
    std::cout << ov::get_openvino_version() << std::endl;

    const std::string model_path = "../../model.onnx";
    size_t batch_size = 1;
    size_t input_width = 224;
    size_t input_height = 224;
    const std::string device_name = "CPU";
    // const std::string device_name = "GPU";
    const int shape[] = { batch_size, 3, input_height, input_width };

    ov::Core core;

    std::cout << "Loading model files: " << model_path << std::endl;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
    OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

    std::string input_tensor_name = model->input().get_any_name();
    //std::string output_tensor_name = model->output().get_any_name();

    ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    const int total_size = shape[0] * shape[1] * shape[2] * shape[3];
    std::shared_ptr<float> image_data(new float[total_size], std::default_delete<float[]>());

    /*std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    float* raw_ptr = image_data.get();
    for (int i = 0; i < total_size; ++i) {
        raw_ptr[i] = dis(gen);
    }*/

    ov::Tensor input_tensor{ ov::element::f32, {batch_size, 3, input_height, input_width}, image_data.get() };
    infer_request.set_tensor(input_tensor_name, input_tensor);

    long long total_time = 0;
    Timer timer;
    for (int i = 0; i < 20; i++) {
        timer.reset();
        infer_request.infer();
        long long elapsed_time = timer.elapsedMilliseconds();
        std::cout << i << " Elapsed time: " << elapsed_time << " ms" << std::endl;
        if (i >= 10) {
            total_time += elapsed_time;
        }
    }
    printf("Running done");
    std::cout << "OpenVINO C++:" << device_name << " Average elapsed time : " << total_time / 10 << " ms" << std::endl;

    std::cout << "Infer done" << std::endl;
    int a;
    std::cin >> a;

    return 0;
}

