/*!
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>

namespace {
std::mt19937 mt(0);
inline void  initVector(std::vector<double>& src)
{
    std::uniform_real_distribution<> rnd01(0.0, 1.0);
    for (auto& s : src) {
        s = rnd01(mt);
    }
}

double cpuSum(const std::vector<double>& src)
{
    double acc = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        acc += src[i];
    }
    return acc;
}

inline size_t getFileSize(std::ifstream& file)
{
    file.seekg(0, std::ios::end);
    size_t ret = file.tellg();
    file.seekg(0, std::ios::beg);

    return ret;
}

inline void loadFile(std::ifstream& file, std::vector<char>& d, size_t size)
{
    d.resize(size);
    file.read(reinterpret_cast<char*>(d.data()), size);
}

cl::Program createProgram(cl::Context& context, const std::vector<cl::Device>& devices, const std::string& filename)
{
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);

    if (file.fail()) {
        throw "can not open kernel file";
    }

    size_t            filesize = getFileSize(file);
    std::vector<char> binary_data;
    loadFile(file, binary_data, filesize);

    cl::Program::Binaries binaries;
    binaries.push_back(std::make_pair(&binary_data[0], filesize));

    return cl::Program(context, devices, binaries, nullptr, nullptr);
}

cl::Program createProgram(cl::Context& context, const cl::Device& device, const std::string& filename)
{
    std::vector<cl::Device> devices { device };
    return createProgram(context, devices, filename);
}

void getBasicDeviceInfo(const cl::Device &device, std::string& device_name, size_t& global_work_size)
{
    device.getInfo(CL_DEVICE_NAME, &device_name);

    size_t global_work_size_[3] = { 0 };
    device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &global_work_size_);

    global_work_size = global_work_size_[0];
    if (device_name.find("PEZY-SC2") != std::string::npos) {
        global_work_size = std::min(global_work_size, (size_t)15872);
    }
}

void benchmarkSum(const std::vector<double>& src)
{
    const size_t loop_count = 20;
    const double expected = cpuSum(src);
    const std::vector<std::string> kernel_names = { "sum_simple", "sum_base2", "sum_base4" };

    try {
        // Get Platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        const auto& Platform = platforms[0];

        // Get devices
        std::vector<cl::Device> devices;
        Platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

        // Use first device.
        const auto& device = devices[0];

        // Create Context.
        auto context = cl::Context(device);

        // Create CommandQueue.
        auto command_queue = cl::CommandQueue(context, device, 0);

        // Create Program.
        // Load compiled binary file and create cl::Program object.
        auto program = createProgram(context, device, "kernel/kernel.pz");

        // Create Buffers.
        size_t num = src.size();
        auto device_src = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
        auto device_dst = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double));

        // Send src.
        command_queue.enqueueWriteBuffer(device_src, true, 0, sizeof(double) * num, &src[0]);

        // Get workitem size.
        // sc1-64: 8192  (1024 PEs * 8 threads)
        // sc2   : 15782 (1984 PEs * 8 threads)
        std::string device_name = "Unknown";
        size_t global_work_size = 0;
        getBasicDeviceInfo(device, device_name, global_work_size);
        std::cout << "Use device : " << device_name << std::endl;
        std::cout << "workitem   : " << global_work_size << std::endl;

        for (const auto& kernel_name : kernel_names) {
            // Create Kernel.
            auto kernel = cl::Kernel(program, kernel_name.c_str());

            // Set kernel args.
            kernel.setArg(0, device_dst);
            kernel.setArg(1, num);
            kernel.setArg(2, device_src);

            double total_time = 0.0; // total elapsed time in nanoseconds
            bool   verify_ok  = true;

            for (size_t i = 0; i < loop_count+1; i++) {
                // Clear dst.
                cl::Event write_event;
                command_queue.enqueueFillBuffer(device_dst, 0, 0, sizeof(double), nullptr, &write_event);
                write_event.wait();

                // Invoke kernel
                auto start = std::chrono::high_resolution_clock::now();
                cl::Event event;
                command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, &event);
                event.wait();
                auto end = std::chrono::high_resolution_clock::now();

                // Check result
                double actual;
                command_queue.enqueueReadBuffer(device_dst, true, 0, sizeof(double), &actual);

                if (std::abs(expected - actual)/std::max(std::abs(expected), std::abs(actual)) > 1e-8) {
                    std::cout << kernel_name << " failed:  expected: " << expected << "   actual: " << actual << std::endl;
                    verify_ok = false;
                    break;
                }

                if (i != 0) {
                    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                }
            }

            // Print result
            if (verify_ok) {
                std::cout << kernel_name << "\t" << (total_time / loop_count)/1e6 << " ms" << std::endl;
            }
        }
    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}
}

int main(int argc, char** argv)
{
    size_t num = 1024;

    if (argc > 1) {
        num = strtol(argv[1], nullptr, 10);
    }

    std::cout << "num " << num << std::endl;

    std::vector<double> src(num);
    initVector(src);

    benchmarkSum(src);

    return 0;
}
