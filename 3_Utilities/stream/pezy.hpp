#ifndef PEZY_HPP
#define PEZY_HPP

#include <cstddef>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <vector>

class pezy {
public:
    pezy(size_t device_id);

    std::vector<std::vector<double>> run(size_t STREAM_ARRAY_SIZE, size_t NTIMES, size_t OFFSET);

private:
    void init(size_t device_id);

    double Empty(void);
    double Copy(cl::Buffer c, cl::Buffer a, size_t num);
    double Scale(cl::Buffer b, cl::Buffer c, double scalar, size_t num);
    double Add(cl::Buffer c, cl::Buffer a, cl::Buffer b, size_t num);
    double Triad(cl::Buffer a, cl::Buffer b, cl::Buffer c, double scalar, size_t num);

    double Kick(cl::Kernel& kernel);

    cl::Context             context;
    cl::CommandQueue        queue;
    std::vector<cl::Kernel> kernels;
    size_t                  global_work_size;
};

#endif
