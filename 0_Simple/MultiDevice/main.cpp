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
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename T>
using pvec = std::vector<std::unique_ptr<T>>;

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

    pvec<cl::Context>      contexts;
    pvec<cl::CommandQueue> queues;
    pvec<cl::Buffer>       buffers;

    for (size_t i = 0; i < M; ++i) {
        auto dev = devs[i];

        // Init Context, Buffers, and Queue
        auto context = new cl::Context(dev);
        auto queue   = new cl::CommandQueue(*context, dev);
        auto buf     = new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(uint32_t) * L);

        // Setup device program. See also the definition in pzc/kernel.pzc
        cl::Program::Binaries bins = { { &pz_binary[0], pz_binary.size() } };
        cl::Program           program(*context, { dev }, bins);

        auto kernel = cl::make_kernel<size_t, cl::Buffer&, uint32_t>(program, "fill");

        size_t work_size = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
        std::clog << "Work size = " << work_size << std::endl;

        kernel(cl::EnqueueArgs(*queue, cl::NDRange(work_size)), N, *buf, value);
        cl::copy(*queue, *buf, a.begin() + i * L, a.end() + (i + 1) * L);

        contexts.emplace_back(context);
        queues.emplace_back(queue);
        buffers.emplace_back(buf);
    }

    for (auto&& queue : queues) {
        queue->finish();
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
