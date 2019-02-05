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

namespace {
// pzc kernel source code for pzc_add
std::string kernel_src_pzc_add = R"(
#include <pzc_builtin.h>

void pzc_add(size_t        num,
             double*       dst,
             const double* src0,
             const double* src1)
{
    size_t       pid              = get_pid();
    size_t       tid              = get_tid();
    size_t       gid              = pid * get_maxtid() + tid;
    const size_t GLOBAL_WORK_SIZE = get_maxtid() * get_maxpid();

    for (size_t i = gid; i < num; i += GLOBAL_WORK_SIZE) {
        double s0 = src0[i];
        double s1 = src1[i];
        chgthread();
        dst[i] = s0 + s1;
    }

    flush();
}
)";
std::mt19937 mt(0);
inline void  initVector(std::vector<double>& src)
{
    std::uniform_real_distribution<> rnd01(0.0, 1.0);
    for (auto& s : src) {
        s = rnd01(mt);
    }
}

void cpuAdd(size_t num, std::vector<double>& dst, const std::vector<double>& src0, const std::vector<double>& src1)
{
    for (size_t i = 0; i < num; ++i) {
        dst[i] = src0[i] + src1[i];
    }
}

void pzcAdd(size_t num, std::vector<double>& dst, const std::vector<double>& src0, const std::vector<double>& src1)
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
        // Online compile on target device with pzc source code.
        auto program = cl::Program(context, kernel_src_pzc_add, nullptr);
        program.build(devices, nullptr, nullptr, nullptr);

        // Create Kernel.
        // Give kernel name without pzc_ prefix.
        auto kernel = cl::Kernel(program, "add");

        // Create Buffers.
        auto device_src0 = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
        auto device_src1 = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
        auto device_dst  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);

        // Send src.
        command_queue.enqueueWriteBuffer(device_src0, true, 0, sizeof(double) * num, &src0[0]);
        command_queue.enqueueWriteBuffer(device_src1, true, 0, sizeof(double) * num, &src1[0]);

        // Clear dst.
        cl::Event write_event;
        command_queue.enqueueFillBuffer(device_dst, 0, 0, sizeof(double) * num, nullptr, &write_event);
        write_event.wait();

        // Set kernel args.
        kernel.setArg(0, num);
        kernel.setArg(1, device_dst);
        kernel.setArg(2, device_src0);
        kernel.setArg(3, device_src1);

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
        command_queue.enqueueReadBuffer(device_dst, true, 0, sizeof(double) * num, &dst[0]);

        // Finish all commands.
        command_queue.flush();
        command_queue.finish();

    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}

bool verify(const std::vector<double>& actual, const std::vector<double>& expected)
{
    assert(actual.size() == expected.size());

    bool   is_true     = true;
    size_t error_count = 0;

    const size_t num = actual.size();
    for (size_t i = 0; i < num; ++i) {
        if (fabs(actual[i] - expected[i]) > 1.e-7) {

            if (error_count < 10) {
                std::cerr << "# ERROR " << i << " " << actual[i] << " " << expected[i] << std::endl;
            }
            error_count++;
            is_true = false;
        }
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

    std::vector<double> src0(num);
    std::vector<double> src1(num);
    initVector(src0);
    initVector(src1);

    std::vector<double> dst_sc(num, 0);
    std::vector<double> dst_cpu(num, 0);

    // run cpu add
    cpuAdd(num, dst_cpu, src0, src1);

    // run device add
    pzcAdd(num, dst_sc, src0, src1);

    // verify
    if (verify(dst_sc, dst_cpu)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }

    return 0;
}
