using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;

namespace OnnxRuntimeCSharpExample
{
    class Program
    {
        enum Device
        {
            Onnxruntime_cpu,
            Onnxruntime_gpu,
            DirectML_A770_16G,  // Microsoft.ML.OnnxRuntime.DirectML
            DirectML_GTX1070Ti,  // Microsoft.ML.OnnxRuntime.DirectML
        }
        static void Main(string[] args)
        {

            int batch_size = 4;
            Device device = Device.Onnxruntime_cpu;
            string modelPath = "model.onnx";

            Console.WriteLine($"batch_size: {batch_size}");


            var options = new SessionOptions();
            // Notice: You need to change NuGet packages first.
            switch (device)
            {
                case Device.Onnxruntime_cpu:
                    break;
                case Device.Onnxruntime_gpu:
                    options.AppendExecutionProvider_CUDA(0);
                    break;
                case Device.DirectML_A770_16G:
                    options.AppendExecutionProvider_DML(0);
                    break;
                case Device.DirectML_GTX1070Ti:
                    options.AppendExecutionProvider_DML(1);
                    break;
                default:
                    break;
            }
            using var session = new InferenceSession(modelPath, options);
            float[] inputData = new float[batch_size * 3 * 224 * 224];

            var tensor = new DenseTensor<float>(inputData, new int[] { batch_size, 3, 224, 224 });
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };

            
            long total_time = 0;
            for (int i = 0; i < 20; i++)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                using var results = session.Run(inputs);  // results 是 IDisposable
                stopwatch.Stop();
                long elapsedMilliseconds = stopwatch.ElapsedMilliseconds;
                Console.WriteLine($"{i}: Elapsed time: {elapsedMilliseconds} ms");
                if (i >= 10)
                {
                    total_time += elapsedMilliseconds;
                }
            }
            total_time /= 10;
            Console.WriteLine($"Average elapsed time: {total_time} ms");
            Console.ReadLine();
        }
    }
}
