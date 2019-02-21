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
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
std::mt19937 mt(0);
inline void  initVector(std::vector<double>& src)
{
    std::uniform_real_distribution<> rnd01(0.0, 1.0);
    for (auto& s : src) {
        s = rnd01(mt);
    }
}

void cpuAtomicAdd(size_t num, const std::vector<double>& src, double& dst)
{
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
#pragma omp atomic update
        dst += src[i];
    }
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

void pzcAtomicAdd(size_t num, const std::vector<double>& src, double& dst)
{
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

        // Create Kernel.
        // Give kernel name without pzc_ prefix.
        auto kernel = cl::Kernel(program, "atomic_add");

        // Create Buffers.
        auto device_src = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
        auto device_dst = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double));

        // Send src.
        command_queue.enqueueWriteBuffer(device_src, true, 0, sizeof(double) * num, &src[0]);

        // Set kernel args.
        kernel.setArg(0, device_src);
        kernel.setArg(1, device_dst);
        kernel.setArg(2, num);

        // Get workitem size.
        // sc1-64: 8192  (1024 PEs * 8 threads)
        // sc2   : 15782 (1984 PEs * 8 threads)
        size_t global_work_size = 0;
        {
            std::string device_name;
            device.getInfo(CL_DEVICE_NAME, &device_name);

            size_t global_work_size_[3] = { 0 };
            device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &global_work_size_);

            global_work_size = global_work_size_[0];
            if (device_name.find("PEZY-SC2") != std::string::npos) {
                global_work_size = std::min(global_work_size, (size_t)15872);
            }

            std::cout << "Use device : " << device_name << std::endl;
            std::cout << "workitem   : " << global_work_size << std::endl;
        }

        // Run device kernel.
        cl::Event event;
        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange, nullptr, &event);

        // Waiting device completion.
        event.wait();

        // Get dst.
        command_queue.enqueueReadBuffer(device_dst, true, 0, sizeof(double), &dst);

        // Finish all commands.
        command_queue.flush();
        command_queue.finish();

    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}

bool verify(const double actual, const double expected)
{
    bool is_true = true;

    if (fabs(actual - expected) > 1.e-7) {
        std::cerr << "# ERROR "
                  << " " << actual << " " << expected << std::endl;
        is_true = false;
    }

    return is_true;
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

    double dst_sc  = 0;
    double dst_cpu = 0;

    // run cpu atomic add
    cpuAtomicAdd(num, src, dst_cpu);

    // run device atomic add
    pzcAtomicAdd(num, src, dst_sc);

    // verify
    if (verify(dst_sc, dst_cpu)) {
        std::cout << "PASS" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL" << std::endl;
        return 1;
    }
}
