#include "pezy.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
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

double empty_kernel_execute_time = 0;

void checkSTREAMresults(const double* a, const double* b, const double* c,
                        size_t STREAM_ARRAY_SIZE, size_t NTIMES)
{
    using STREAM_TYPE = double;

    STREAM_TYPE aj, bj, cj, scalar;
    STREAM_TYPE aSumErr, bSumErr, cSumErr;
    STREAM_TYPE aAvgErr, bAvgErr, cAvgErr;
    double      epsilon;
    int         ierr, err;

    /* reproduce initialization */
    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    /* a[] is modified during timing check */
    aj = 2.0E0 * aj;
    /* now execute timing loop */
    scalar = 3.0;
    for (size_t k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    /* accumulate deltas between observed and expected results */
    aSumErr = 0.0;
    bSumErr = 0.0;
    cSumErr = 0.0;
    for (size_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
        aSumErr += abs(a[j] - aj);
        bSumErr += abs(b[j] - bj);
        cSumErr += abs(c[j] - cj);
        // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
    }
    aAvgErr = aSumErr / (STREAM_TYPE)STREAM_ARRAY_SIZE;
    bAvgErr = bSumErr / (STREAM_TYPE)STREAM_ARRAY_SIZE;
    cAvgErr = cSumErr / (STREAM_TYPE)STREAM_ARRAY_SIZE;

    if (sizeof(STREAM_TYPE) == 4) {
        epsilon = 1.e-6;
    } else if (sizeof(STREAM_TYPE) == 8) {
        epsilon = 1.e-13;
    } else {
        printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n", sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    err = 0;
    if (abs(aAvgErr / aj) > epsilon) {
        err++;
        printf("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n", epsilon);
        printf("	  Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", aj, aAvgErr, abs(aAvgErr) / aj);
        ierr = 0;
        for (size_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
            if (abs(a[j] / aj - 1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("		 array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                           j, aj, a[j], abs((aj - a[j]) / aAvgErr));
                }
#endif
            }
        }
        printf("	 For array a[], %d errors were found.\n", ierr);
    }
    if (abs(bAvgErr / bj) > epsilon) {
        err++;
        printf("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n", epsilon);
        printf("	  Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", bj, bAvgErr, abs(bAvgErr) / bj);
        printf("	  AvgRelAbsErr > Epsilon (%e)\n", epsilon);
        ierr = 0;
        for (size_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
            if (abs(b[j] / bj - 1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("		 array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                           j, bj, b[j], abs((bj - b[j]) / bAvgErr));
                }
#endif
            }
        }
        printf("	 For array b[], %d errors were found.\n", ierr);
    }
    if (abs(cAvgErr / cj) > epsilon) {
        err++;
        printf("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n", epsilon);
        printf("	  Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", cj, cAvgErr, abs(cAvgErr) / cj);
        printf("	  AvgRelAbsErr > Epsilon (%e)\n", epsilon);
        ierr = 0;
        for (size_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
            if (abs(c[j] / cj - 1.0) > epsilon) {
                ierr++;
#ifdef VERBOSE
                if (ierr < 10) {
                    printf("		 array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
                           j, cj, c[j], abs((cj - c[j]) / cAvgErr));
                }
#endif
            }
        }
        printf("	 For array c[], %d errors were found.\n", ierr);
    }
    if (err == 0) {
        printf("Solution Validates: avg error less than %e on all three arrays\n", epsilon);
    }
#ifdef VERBOSE
    printf("Results Validation Verbose Results: \n");
    printf("	 Expected a(1), b(1), c(1): %f %f %f \n", aj, bj, cj);
    printf("	 Observed a(1), b(1), c(1): %f %f %f \n", a[1], b[1], c[1]);
    printf("	 Rel Errors on a, b, c:		%e %e %e \n", abs(aAvgErr / aj), abs(bAvgErr / bj), abs(cAvgErr / cj));
#endif
}
}

pezy::pezy(size_t device_id)
{
    init(device_id);
}

void pezy::init(size_t device_id)
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
            std::cerr << "Invalid device id. Use first device." << std::endl;
            device_id = 0;
        }

        const auto& device = devices[device_id];

        context = cl::Context(device);
        queue   = cl::CommandQueue(context, device, 0);

        auto program = createProgram(context, device, "kernel/kernel.pz");

        kernels.push_back(cl::Kernel(program, "Empty"));
        kernels.push_back(cl::Kernel(program, "Copy"));
        kernels.push_back(cl::Kernel(program, "Scale"));
        kernels.push_back(cl::Kernel(program, "Add"));
        kernels.push_back(cl::Kernel(program, "Triad"));

        typedef CL_API_ENTRY          pzcl_int(CL_API_CALL * pfnPezyExtSetCacheWriteBuffer)(pzcl_context context, size_t index, bool enable);
        pfnPezyExtSetCacheWriteBuffer clExtSetCacheWriteBuffer = (pfnPezyExtSetCacheWriteBuffer)clGetExtensionFunctionAddress("pezy_set_cache_writebuffer");
        if (!clExtSetCacheWriteBuffer) {
            throw cl::Error(-1, "clGetExtensionFunctionAddress: Can not get pezy_set_cache_writebuffer");
        }

        if ((clExtSetCacheWriteBuffer(context(), 0, CL_TRUE)) != CL_SUCCESS) {
            throw cl::Error(-1, "clExtSetCacheWriteBuffer failed");
        }

        // Get global work size.
        // sc1-64: 8192  (1024 PEs * 8 threads)
        // sc2   : 15782 (1984 PEs * 8 threads)
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
    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }
}

std::vector<std::vector<double>> pezy::run(size_t STREAM_ARRAY_SIZE, size_t NTIMES, size_t OFFSET)
{
    std::vector<std::vector<double>> times(NTIMES);
    for (auto& t : times) {
        t.resize(4);
    }

    // create buffer
    size_t  allocate_num = STREAM_ARRAY_SIZE + OFFSET;
    double* h_a          = new double[allocate_num];
    double* h_b          = new double[allocate_num];
    double* h_c          = new double[allocate_num];

    std::fill(h_a, h_a + allocate_num, 1.0);
    std::fill(h_b, h_b + allocate_num, 2.0);
    std::fill(h_c, h_c + allocate_num, 0.0);

    for (size_t i = 0; i < allocate_num; ++i) {
        h_a[i] *= 2.0;
    }

    double scalar = 3.0;

    try {
        // empty kernel run
        for (size_t i = 0; i < NTIMES; ++i) {
            empty_kernel_execute_time += Empty();
        }
        empty_kernel_execute_time /= static_cast<double>(NTIMES);

        // create device buffer & write
        auto d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * allocate_num);
        auto d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * allocate_num);
        auto d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * allocate_num);

        queue.enqueueWriteBuffer(d_a, true, 0, sizeof(double) * allocate_num, h_a);
        queue.enqueueWriteBuffer(d_b, true, 0, sizeof(double) * allocate_num, h_b);
        queue.enqueueWriteBuffer(d_c, true, 0, sizeof(double) * allocate_num, h_c);

        for (size_t i = 0; i < NTIMES; ++i) {
            times[i][0] = Copy(d_c, d_a, STREAM_ARRAY_SIZE);
            times[i][1] = Scale(d_b, d_c, scalar, STREAM_ARRAY_SIZE);
            times[i][2] = Add(d_c, d_a, d_b, STREAM_ARRAY_SIZE);
            times[i][3] = Triad(d_a, d_b, d_c, scalar, STREAM_ARRAY_SIZE);
        }

        queue.enqueueReadBuffer(d_a, true, 0, sizeof(double) * allocate_num, h_a);
        queue.enqueueReadBuffer(d_b, true, 0, sizeof(double) * allocate_num, h_b);
        queue.enqueueReadBuffer(d_c, true, 0, sizeof(double) * allocate_num, h_c);
    } catch (const cl::Error& e) {
        std::stringstream msg;
        msg << "CL Error : " << e.what() << " " << e.err();
        throw std::runtime_error(msg.str());
    }

    // verify
    checkSTREAMresults(h_a, h_b, h_c, STREAM_ARRAY_SIZE, NTIMES);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return times;
}

double pezy::Kick(cl::Kernel& kernel)
{
    auto start = std::chrono::high_resolution_clock::now();

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NDRange(), nullptr, &event);
    event.wait();

    auto                                      end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);

    return dur.count() / 1000.0;
}

double pezy::Empty()
{
    auto& kernel = kernels[0];
    return Kick(kernel);
}

double pezy::Copy(cl::Buffer c, cl::Buffer a, size_t num)
{
    auto& kernel = kernels[1];

    kernel.setArg(0, c);
    kernel.setArg(1, a);
    kernel.setArg(2, num);

    return Kick(kernel);
}

double pezy::Scale(cl::Buffer b, cl::Buffer c, double scalar, size_t num)
{
    auto& kernel = kernels[2];

    kernel.setArg(0, b);
    kernel.setArg(1, c);
    kernel.setArg(2, scalar);
    kernel.setArg(3, num);

    return Kick(kernel);
}

double pezy::Add(cl::Buffer c, cl::Buffer a, cl::Buffer b, size_t num)
{
    auto& kernel = kernels[3];

    kernel.setArg(0, c);
    kernel.setArg(1, a);
    kernel.setArg(2, b);
    kernel.setArg(3, num);

    return Kick(kernel);
}

double pezy::Triad(cl::Buffer a, cl::Buffer b, cl::Buffer c, double scalar, size_t num)
{
    auto& kernel = kernels[4];

    kernel.setArg(0, a);
    kernel.setArg(1, b);
    kernel.setArg(2, c);
    kernel.setArg(3, scalar);
    kernel.setArg(4, num);

    return Kick(kernel);
}
