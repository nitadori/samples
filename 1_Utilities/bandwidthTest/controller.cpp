#include "controller.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

typedef CL_API_ENTRY cl_int(CL_API_CALL* PezyExtMemLock)(cl_context, void*, size_t);
typedef CL_API_ENTRY cl_int(CL_API_CALL* PezyExtMemUnLock)(cl_context, void*, size_t);

namespace {
constexpr size_t DEFAULT_SIZE = (32 * (1 << 20));

std::mt19937 mt(0);

template <typename T>
void fill(std::vector<T>& vec)
{
    for (auto& v : vec) {
        v = mt();
    }
}

PezyExtMemLock   clExtMemLock   = nullptr;
PezyExtMemUnLock clExtMemUnLock = nullptr;

void lockFunc(cl::Context& context, void* ptr, size_t size)
{
    cl_int ret = clExtMemLock(context(), ptr, size);
    if (ret != CL_SUCCESS) {
        throw cl::Error(-2, "clExtMemLock: Can not lock mem");
    }
}

void unLockFunc(cl::Context& context, void* ptr, size_t size)
{
    cl_int ret = clExtMemUnLock(context(), ptr, size);
    if (ret != CL_SUCCESS) {
        throw cl::Error(-3, "clExtMemUnLock: Can not unlock mem");
    }
}

void checkAndLock(pezy::MEMMODE mem_mode, cl::Context& context, void* ptr, size_t size)
{
    // Check pinned or pageable
    if (mem_mode == pezy::PINNED) {
        lockFunc(context, ptr, size);
    }
}

void checkAndUnLock(pezy::MEMMODE mem_mode, cl::Context& context, void* ptr, size_t size)
{
    if (mem_mode == pezy::PINNED) {
        unLockFunc(context, ptr, size);
    }
}

void dispTrans()
{
    std::cout << "\tTransfer Size(byte)\t\tBandwidth(MB/s)" << std::endl;
}

void dispMemMode(pezy::MEMMODE mem_mode)
{
    if (mem_mode == pezy::PAGEABLE) {
        std::cout << " PAGEABLE Memory Transfers " << std::endl;
    } else if (mem_mode == pezy::PINNED) {
        std::cout << " PINNED Memory Transfers " << std::endl;
    } else {
    }
}

bool verify(const std::vector<size_t>& actual, std::vector<size_t>& expected)
{
    assert(actual.size() == expected.size());
    size_t num = actual.size();

    bool   is_true     = true;
    size_t error_count = 0;
    for (size_t i = 0; i < num; ++i) {
        if (actual[i] != expected[i]) {
            is_true = false;
            if (error_count < 10) {
                std::cerr << i << ": " << actual[i] << " " << expected[i] << std::endl;
                error_count++;
            }
        }
    }
    return is_true;
}
}

namespace pezy {
Controller::Controller(size_t id_)
    : device_id(id_)
    , loop_count(10)
{
    init();
}

void Controller::init()
{
    try {
        // Get Platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        const auto& Platform = platforms[0];

        // Get devices
        std::vector<cl::Device> devices;
        Platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

        if (device_id > devices.size()) {
            std::cerr << "Invalid device id. Use first device " << std::endl;
            device_id = 0;
        }

        const auto& device = devices[device_id];
        context            = cl::Context(device);

        cl_command_queue_properties prop = 0;
        prop                             = CL_QUEUE_PROFILING_ENABLE;
        queue                            = cl::CommandQueue(context, device, prop);

        // get memlock function
        clExtMemLock = (PezyExtMemLock)clGetExtensionFunctionAddress("pezy_mem_lock");
        if (clExtMemLock == nullptr) {
            throw cl::Error(-1, "clGetExtensitonFunctionAddress: Can not get pezy_mem_lock");
        }

        clExtMemUnLock = (PezyExtMemUnLock)clGetExtensionFunctionAddress("pezy_mem_unlock");
        if (clExtMemUnLock == nullptr) {
            throw cl::Error(-1, "clGetExtensitonFunctionAddress: Can not get pezy_mem_unlock");
        }
    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}

void Controller::showDeviceInfo() const
{
    try {
        cl::Device device;
        context.getInfo(CL_CONTEXT_DEVICES, &device);

        std::string device_name;
        device.getInfo(CL_DEVICE_NAME, &device_name);

        std::cout << " Device " << device_id << " " << device_name << std::endl;

    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}

void Controller::runTest(const param_t& params)
{
    std::cout << "Bandwidth test start..." << std::endl;
    showDeviceInfo();

    switch (params.mode) {
    case QUICK:
        std::cout << " Quick Mode" << std::endl;
        test(params.mem_mode, params.measure, DEFAULT_SIZE, DEFAULT_SIZE, DEFAULT_SIZE);
        break;
    case RANGE:
        std::cout << " Range Mode" << std::endl;
        test(params.mem_mode, params.measure, params.range_start, params.range_end, params.range_inc);
        break;
    }
}

void Controller::testOneShot(void* ptr, cl::Buffer& buf, size_t size, std::function<void(cl::Buffer&, size_t, void*)> trans)
{
    double elap = 0; // nano

    for (size_t i = 0; i < loop_count; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        trans(buf, size, ptr);

        auto end = std::chrono::high_resolution_clock::now();

        elap += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    double e   = elap / (double)loop_count;
    double mbs = size / e * 1.e3;

    //std::cout << "\t" << size << "\t\t\t\t" << mbs << std::endl;
    std::cout << "\t" << size << "\t\t\t";
    if (std::to_string(size).length() < 8)
        std::cout << "\t";
    std::cout << mbs << std::endl;
}

void Controller::test(MEMMODE mem_mode, MEASURE measure, size_t range_start, size_t range_end, size_t range_inc)
{
    bool is_true = true;

    if (measure == HtoD || measure == ALL) {
        std::cout << "\n";
        std::cout << " Host to Device Bandwidth, " << std::endl;
        dispMemMode(mem_mode);

        // Host to device
        auto trans = [this](cl::Buffer& buf, size_t size, void* ptr) {
            cl::Event event;
            queue.enqueueWriteBuffer(buf, false, 0, size, ptr, nullptr, &event);
            event.wait();
        };

        dispTrans();

        for (size_t i = range_start; i <= range_end; i += range_inc) {
            size_t              size = (i + (sizeof(size_t) - 1)) & ~(sizeof(size_t) - 1);
            std::vector<size_t> host_src(size / sizeof(size_t));
            fill(host_src);

            void* host_src_ptr = &host_src[0];
            checkAndLock(mem_mode, context, host_src_ptr, size);

            auto buf = cl::Buffer(context, CL_MEM_READ_WRITE, size);

            testOneShot(host_src_ptr, buf, size, trans);

            // check
            std::vector<size_t> host_dst(size / sizeof(size_t));
            queue.enqueueReadBuffer(buf, true, 0, size, &host_dst[0]);

            checkAndUnLock(mem_mode, context, host_src_ptr, size);

            if (!verify(host_dst, host_src)) {
                std::cerr << " " << size << " Write Test failed " << std::endl;
                is_true = false;
            }
        }
    }

    if (measure == DtoH || measure == ALL) {
        std::cout << "\n";
        std::cout << " Device to Host Bandwidth, " << std::endl;
        dispMemMode(mem_mode);

        // Device to Host
        auto trans = [this](cl::Buffer& buf, size_t size, void* ptr) {
            cl::Event event;
            queue.enqueueReadBuffer(buf, false, 0, size, ptr, nullptr, &event);
            event.wait();
        };

        dispTrans();
        for (size_t i = range_start; i <= range_end; i += range_inc) {
            size_t              size = (i + (sizeof(size_t) - 1)) & ~(sizeof(size_t) - 1);
            std::vector<size_t> host_src(size / sizeof(size_t));
            fill(host_src);

            void* host_src_ptr = &host_src[0];
            checkAndLock(mem_mode, context, host_src_ptr, size);

            auto buf = cl::Buffer(context, CL_MEM_READ_WRITE, size);
            queue.enqueueWriteBuffer(buf, true, 0, size, host_src_ptr);

            std::vector<size_t> host_dst(size / sizeof(size_t));
            void*               host_dst_ptr = &host_dst[0];
            checkAndLock(mem_mode, context, host_dst_ptr, size);

            testOneShot(host_dst_ptr, buf, size, trans);

            // check
            checkAndUnLock(mem_mode, context, host_src_ptr, size);
            checkAndUnLock(mem_mode, context, host_dst_ptr, size);

            if (!verify(host_dst, host_src)) {
                std::cerr << " " << size << " Read Test failed " << std::endl;
                is_true = false;
            }
        }
    }

    // verify
    std::cout << "\n";
    std::cout << "RESULT = ";
    if (is_true) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

} // namespace pezy
