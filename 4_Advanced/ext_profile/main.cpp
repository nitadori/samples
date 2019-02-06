/*!
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
// Please refer to the pzcl_ext.h or Runtime Extensions section in pzsdk document.
pfnPezyExtSetProfile                clExtSetProfile                = nullptr;
pfnPezyExtGetProfilePEStatistics    clExtGetProfilePEStatistics    = nullptr;
pfnPezyExtGetProfilePE              clExtGetProfilePE              = nullptr;
pfnPezyExtGetProfileCacheStatistics clExtGetProfileCacheStatistics = nullptr;
pfnPezyExtGetProfileCache           clExtGetProfileCache           = nullptr;
std::mt19937                        mt(0);
inline void                         initVector(std::vector<double>& src)
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

void initExtension()
{
    // get the extension function addresss.
    clExtSetProfile                = (pfnPezyExtSetProfile)clGetExtensionFunctionAddress("pezy_set_profile");
    clExtGetProfilePEStatistics    = (pfnPezyExtGetProfilePEStatistics)clGetExtensionFunctionAddress("pezy_get_profile_pe_statistics");
    clExtGetProfilePE              = (pfnPezyExtGetProfilePE)clGetExtensionFunctionAddress("pezy_get_profile_pe");
    clExtGetProfileCacheStatistics = (pfnPezyExtGetProfileCacheStatistics)clGetExtensionFunctionAddress("pezy_get_profile_cache_statistics");
    clExtGetProfileCache           = (pfnPezyExtGetProfileCache)clGetExtensionFunctionAddress("pezy_get_profile_cache");

    if (!clExtSetProfile || !clExtGetProfilePEStatistics || !clExtGetProfilePE || !clExtGetProfileCacheStatistics || !clExtGetProfileCache) {
        throw "can not get extension function pointer(s)";
    }
}

void showProfile(cl::Context& context, size_t global_work_size)
{
    cl_int       ret              = CL_SUCCESS;
    const size_t threads_per_city = 128; // use global_work_size if you want to see all profiles.

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "***** PE profile statistics *****" << std::endl;
    {
        // Show the PE profile statistics.
        pzcl_profile_pe_stats stats = { 0 };
        stats.size                  = sizeof(pzcl_profile_pe_stats);
        ret                         = clExtGetProfilePEStatistics(context(), 0, &stats);
        if (CL_SUCCESS != ret)
            throw cl::Error(ret, "clExtGetProfilePEStatistics returns error");

        std::cout << " elapse_ns  : " << stats.elapse_ns << " [ns]" << std::endl;
        std::cout << " efficiency : " << stats.efficiency << " [%]" << std::endl; // (run - (stall + wait)) / run
    }

    std::cout << "***** First city PE profile *****" << std::endl;
    {
        // Show the PE profile in first city.
        // +---------+----+-------+----+------+
        // |         |City|Village| PE |thread|
        // +---------+----+-------+----+------+
        // | City    |  1 |     4 | 16 |  128 |
        // | Village |    |     1 |  4 |   32 |
        // | PE      |    |       |  1 |    8 |
        // +---------+----+-------+----+------+
        // 1 city has 128 threads.
        // 1 PE   has   8 threads.
        // 1 city has 128/8 PEs.
        size_t per_PE_threads = 8;
        size_t pe_count       = threads_per_city / per_PE_threads;
        for (size_t i = 0; i < pe_count; i++) {
            pzcl_profile_pe profile = { 0 };
            ret                     = clExtGetProfilePE(context(), 0, i, &profile);
            if (CL_SUCCESS != ret)
                throw cl::Error(ret, "clExtGetProfilePE returns error");

            std::cout << " PE"
                      << std::setw(4) << i << " (run, stall, wait) : ("
                      << std::setw(5) << profile.run << ","
                      << std::setw(5) << profile.stall << ","
                      << std::setw(5) << profile.wait << ")"
                      << " [cycle] "
                      << std::endl;
        }
    }

    std::cout << "***** L1 cache profile statistics *****" << std::endl;
    {
        pzcl_profile_cache_stats stats = { 0 };
        stats.size                     = sizeof(pzcl_profile_cache_stats);
        ret                            = clExtGetProfileCacheStatistics(context(), 0, PZCL_EXT_PROFILE_CACHE_L1, &stats);
        if (CL_SUCCESS != ret)
            throw cl::Error(ret, "clExtGetProfileCacheStatistics returns error");

        std::cout << " read  hit rate : " << stats.read_hit_rate << " [%]" << std::endl;
        std::cout << " write hit rate : " << stats.write_hit_rate << " [%]" << std::endl;
    }

    std::cout << "***** First city L1 cache profile *****" << std::endl;
    {
        // L1 has  32 threads (     4 * 8 threads)
        size_t per_cache_threads = 32;
        size_t cache_count       = threads_per_city / per_cache_threads;
        for (size_t i = 0; i < cache_count; i++) {
            pzcl_profile_cache profile = { 0 };
            ret                        = clExtGetProfileCache(context(), 0, PZCL_EXT_PROFILE_CACHE_L1, i, &profile);
            if (CL_SUCCESS != ret)
                throw cl::Error(ret, "clExtGetProfileCache returns error");

            std::cout << " Vill"
                      << std::setw(3) << i << " read  (request, hit) : ("
                      << std::setw(4) << profile.read_request << ","
                      << std::setw(4) << profile.read_hit << ")"
                      << " [count] "
                      << std::endl;
            std::cout << " Vill"
                      << std::setw(3) << i << " write (request, hit) : ("
                      << std::setw(4) << profile.write_request << ","
                      << std::setw(4) << profile.write_hit << ")"
                      << " [count] "
                      << std::endl;
        }
    }

    std::cout << "***** L2 cache profile statistics *****" << std::endl;
    {
        pzcl_profile_cache_stats stats = { 0 };
        stats.size                     = sizeof(pzcl_profile_cache_stats);
        ret                            = clExtGetProfileCacheStatistics(context(), 0, PZCL_EXT_PROFILE_CACHE_L2, &stats);
        if (CL_SUCCESS != ret)
            throw cl::Error(ret, "clExtGetProfileCacheStatistics returns error");

        std::cout << " read  hit rate : " << stats.read_hit_rate << " [%]" << std::endl;
        std::cout << " write hit rate : " << stats.write_hit_rate << " [%]" << std::endl;
    }

    std::cout << "***** First city L2 cache profile *****" << std::endl;
    {
        // L2 has 128 threads ( 4 * 4 * 8 threads)
        size_t per_cache_threads = 128;
        size_t cache_count       = threads_per_city / per_cache_threads;
        for (size_t i = 0; i < cache_count; i++) {
            pzcl_profile_cache profile = { 0 };
            ret                        = clExtGetProfileCache(context(), 0, PZCL_EXT_PROFILE_CACHE_L2, i, &profile);
            if (CL_SUCCESS != ret)
                throw cl::Error(ret, "clExtGetProfileCache returns error");

            std::cout << " City"
                      << std::setw(3) << i << " read  (request, hit) : ("
                      << std::setw(4) << profile.read_request << ","
                      << std::setw(4) << profile.read_hit << ")"
                      << " [count] "
                      << std::endl;
            std::cout << " City"
                      << std::setw(3) << i << " write (request, hit) : ("
                      << std::setw(4) << profile.write_request << ","
                      << std::setw(4) << profile.write_hit << ")"
                      << " [count] "
                      << std::endl;
        }
    }

    // Limitation:
    // Current pzsdk does not support the EXT_PROFILE_CACHE for LLC on SC2 yet.
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

        // Get extension function pointers.
        initExtension();

        // Enable the profiling for first device.
        clExtSetProfile(context(), 0, CL_TRUE);

        // Create CommandQueue.
        auto command_queue = cl::CommandQueue(context, device, 0);

        // Create Program.
        // Load compiled binary file and create cl::Program object.
        auto program = createProgram(context, device, "kernel/kernel.pz");

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

        // Show the profile result using extension.
        showProfile(context, global_work_size);

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
