//#include <sys/stat.h>

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
//#ifde/*f _WIN32
//#    include "samples/os/windows/w_dirent.h"
//#else
//#    include <dirent.h>
//#endif*/

#include "openvino/openvino.hpp"


using namespace ov::preprocess;

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

int main(int argc, char* argv[]) {
    // -------- Get OpenVINO runtime version --------
    std::cout << ov::get_openvino_version() << std::endl;

    const std::string model_path = "../../model.onnx";
    size_t batch_size = 512;
    size_t input_width = 224;
    size_t input_height = 224;
    //const std::string device_name = "CPU";
    const std::string device_name = "GPU.0";
    //const std::string device_name = "GPU.1";
    const int shape[] = { batch_size, 3, input_height, input_width };

    // -----------------------------------------------------------------------------------------------------

    // -------- Step 1. Initialize OpenVINO Runtime Core ---------
    ov::Core core;

    // -------- Step 2. Read a model --------
    std::cout << "Loading model files: " << model_path << std::endl;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);


    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
    OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

    std::string input_tensor_name = model->input().get_any_name();
    std::string output_tensor_name = model->output().get_any_name();

    //// -------- Step 3. Configure preprocessing  --------
    //PrePostProcessor ppp = PrePostProcessor(model);

    //// 1) Select input with 'input_tensor_name' tensor name
    //InputInfo& input_info = ppp.input(input_tensor_name);

    //// 2) Set input type
    //input_info.tensor()
    //    .set_element_type(ov::element::f32)
    //    .set_color_format(ColorFormat::BGR)
    //    .set_spatial_static_shape(input_height, input_width);
    //// 4) Set model data layout (Assuming model accepts images in NCHW layout)
    //input_info.model().set_layout("NCHW");

    //// 5) Apply preprocessing to an input with 'input_tensor_name' name of loaded model
    //model = ppp.build();

    // -------- Step 4. Loading a model to the device --------
    ov::CompiledModel compiled_model = core.compile_model(model, device_name);

    // -------- Step 5. Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Step 6. Prepare input data  --------
    const int total_size = shape[0] * shape[1] * shape[2] * shape[3];  // 计算总大小

    std::shared_ptr<float> image_data(new float[total_size], std::default_delete<float[]>());  // 分配内存

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 填充随机数据
    float* raw_ptr = image_data.get();
    for (int i = 0; i < total_size; ++i) {
        raw_ptr[i] = dis(gen);  // 生成0到1之间的随机数
    }

    ov::Tensor input_tensor{ ov::element::f32, {batch_size, 3, input_height, input_width}, image_data.get() };

    // -------- Step 6. Set input tensor  --------
    // Set the input tensor by tensor name to the InferRequest
    infer_request.set_tensor(input_tensor_name, input_tensor);

    // -------- Step 7. Do inference --------
    // Running the request synchronously
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
    std::cout << "OpenVINO:" << device_name << " Average elapsed time : " << total_time / 10 << " ms" << std::endl;

    std::cout << "Infer done" << std::endl;
    int a;
    std::cin >> a;

    return EXIT_SUCCESS;
}

