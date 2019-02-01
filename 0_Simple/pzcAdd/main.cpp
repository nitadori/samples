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
std::mt19937 mt(0);
inline void  initVector(std::vector<double>& src)
{
    std::uniform_real_distribution<> rnd01(0.0, 1.0);
    for (auto& s : src) {
        s = rnd01(mt);
    }
}

void cpuAdd(size_t num, const std::vector<double>& src, std::vector<double>& dst)
{
    for (size_t i = 0; i < num; ++i) {
        dst[i] += src[i];
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
    std::vector<cl::Device> devices{ device };
    return createProgram(context, devices, filename);
}

void pzcAdd(size_t num, const std::vector<double>& src, std::vector<double>& dst)
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
        auto kernel = cl::Kernel(program, "add");

        // Create Buffers.
        auto device_src = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
        auto device_dst = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);

        // Send src.
        command_queue.enqueueWriteBuffer(device_src, true, 0, sizeof(double) * num, &src[0]);

        // Clear dst.
        cl::Event write_event;
        command_queue.enqueueFillBuffer(device_dst, 0, 0, sizeof(double) * num, nullptr, &write_event);
        write_event.wait();

        // Set kernel args.
        kernel.setArg(0, device_src);
        kernel.setArg(1, device_dst);
        kernel.setArg(2, num);

        // Run device kernel.
        size_t    global_work_size = 15872;
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

    std::vector<double> src(num);
    initVector(src);

    std::vector<double> dst_sc(num, 0);
    std::vector<double> dst_cpu(num, 0);

    // run cpu add
    cpuAdd(num, src, dst_cpu);

    // run device add
    pzcAdd(num, src, dst_sc);

    // verify
    if (verify(dst_sc, dst_cpu)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }

    return 0;
}
