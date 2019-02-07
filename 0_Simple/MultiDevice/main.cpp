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

std::vector<unsigned char> read_pz_binary()
{
    std::string filename = "kernel/kernel.pz";

    std::ifstream              ifs(filename, std::ios::binary);
    std::vector<unsigned char> bin((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (bin.empty()) {
        throw std::runtime_error("Cannot open PZ Binary");
    }
    return bin;
}

void fill_multi_device(std::vector<uint32_t>& a, uint32_t value)
{
    const size_t N = a.size();

    // Init platform
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        throw std::runtime_error("No platform found");
    }
    cl::Platform plat = platforms[0];
    std::clog << "Using platform: " << plat.getInfo<CL_PLATFORM_NAME>()
              << std::endl;

    // Init device
    std::vector<cl::Device> devs;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &devs);
    if (devs.size() == 0) {
        throw std::runtime_error("No devices found");
    }

    const size_t M = devs.size();
    std::clog << "Use " << M << " device(s)" << std::endl;

    assert(N % M == 0); // Consider M|N case for simplicity
    const size_t L = N / M;

    const auto pz_binary = read_pz_binary();

    std::vector<cl::Context>      contexts;
    std::vector<cl::CommandQueue> queues;
    std::vector<cl::Buffer>       buffers;

    for (size_t i = 0; i < M; ++i) {
        auto dev = devs[i];

        // Init Context, Buffers, and Queue
        cl::Context      context(dev);
        cl::CommandQueue queue(context, dev);
        cl::Buffer       buf(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * L);

        // Setup device program. See also the definition in pzc/kernel.pzc
        cl::Program::Binaries bins = { { &pz_binary[0], pz_binary.size() } };
        cl::Program           program(context, { dev }, bins);

        auto kernel = cl::make_kernel<size_t, cl::Buffer&, uint32_t>(program, "fill");

        size_t work_size = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
        std::clog << "Work size = " << work_size << std::endl;

        kernel(cl::EnqueueArgs(queue, cl::NDRange(work_size)), N, buf, value);
        cl::copy(queue, buf, a.begin() + i * L, a.end() + (i + 1) * L);

        contexts.push_back(std::move(context));
        queues.push_back(std::move(queue));
        buffers.push_back(std::move(buf));
    }

    for (auto&& queue : queues) {
        queue.finish();
    }
}

int main()
{
    const int      N     = 300;
    const uint32_t value = 1234;

    // array to be filled
    std::vector<uint32_t> a(N, 0);

    fill_multi_device(a, value);

    // Check the array is filled
    for (auto v : a) {
        assert(v == value);
    }
    return 0;
}
