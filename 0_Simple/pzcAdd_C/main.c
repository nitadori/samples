/*!
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 */

#define _POSIX_SOURCE
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <pzcl/pzcl_ocl_wrapper.h>

static void initVector(size_t num, double* dst)
{
    srand(8009UL);
    for (size_t i = 0; i < num; i++) {
        dst[i] = rand() / (double)RAND_MAX;
    }
}

static void cpuAdd(size_t num, double* dst, const double* src0, const double* src1)
{
    for (size_t i = 0; i < num; ++i) {
        dst[i] = src0[i] + src1[i];
    }
}

static bool loadFile(const char* path, unsigned char** data, size_t* size)
{
    bool           result    = false;
    int            fd        = -1;
    FILE*          fp        = NULL;
    unsigned char* buf       = NULL;
    size_t         read_size = 0;
    size_t         file_size = 0;
    struct stat    stbuf;

    // init
    *data = NULL;
    *size = 0;

    // get file size and allocate buffer
    fd = open(path, O_RDONLY);
    if (fd == -1)
        goto Leave;

    fp = fdopen(fd, "r");
    if (fp == NULL)
        goto Leave;

    if (fstat(fd, &stbuf) == -1)
        goto Leave;

    file_size = stbuf.st_size;
    buf       = malloc(file_size);
    if (buf == NULL)
        goto Leave;

    // read file
    read_size = fread(buf, 1, file_size, fp);
    if (read_size != file_size)
        goto Leave;

    *data  = buf;
    *size  = file_size;
    result = true;

Leave:
    if (!result)
        free(buf);

    if (fp) {
        fclose(fp);
    } else if (fd >= 0) {
        close(fd);
    }

    return result;
}

static cl_program createProgram(cl_context context, cl_device_id device, const char* path, cl_int* err)
{
    unsigned char* binary_data;
    size_t         file_size;

    if (!loadFile(path, &binary_data, &file_size)) {
        fprintf(stderr, "cannot open kernel file\n");
        return NULL;
    }
    const unsigned char* const_binary_data = binary_data; // for suppressing warnings

    cl_program program = clCreateProgramWithBinary(context, 1, &device, &file_size, &const_binary_data, NULL, err);

    free(binary_data);

    return program;
}

static cl_int executeKernel(cl_device_id device, cl_command_queue queue, cl_kernel kernel)
{
    enum { Max_Device_Name_Size = 256 };
    char     device_name[Max_Device_Name_Size] = { 0 };
    size_t   global_work_size                  = 0;
    cl_event event                             = NULL;

    // Get workitem size.
    // sc1-64: 8192  (1024 PEs * 8 threads)
    // sc2   : 15782 (1984 PEs * 8 threads)
    clGetDeviceInfo(device, CL_DEVICE_NAME, Max_Device_Name_Size - 1, device_name, NULL);
    {
        size_t global_work_size_[3] = { 0 };
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(global_work_size_), global_work_size_, NULL);
        global_work_size = global_work_size_[0];
        if (strstr(device_name, "PEZY-SC2") != NULL && global_work_size > 1984 * 8) {
            global_work_size = 1984 * 8;
        }
    }
    printf("Device   : %s\n", device_name);
    printf("Workitem : %zu\n", global_work_size);

    // Execute kernel
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event);

    // Wait for completion
    if (err == CL_SUCCESS) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }

    return err;
}

static void pzcAdd(const size_t num, double* dst, const double* src0, const double* src1)
{
    cl_int           err;
    cl_platform_id   platform_id = NULL;
    cl_device_id     device_id   = NULL;
    cl_context       context     = NULL;
    cl_program       program     = NULL;
    cl_command_queue queue       = NULL;
    cl_kernel        kernel      = NULL;
    cl_mem           mem_dst     = NULL;
    cl_mem           mem_src0    = NULL;
    cl_mem           mem_src1    = NULL;
    cl_event         write_event = NULL;
    const double     zero        = 0.0;

    // Get Platform
    if ((err = clGetPlatformIDs(1, &platform_id, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        goto Leave;
    }

    // Get devices and use first one
    if ((err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", err);
        goto Leave;
    }

    // Create Context
    if ((context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err)) == NULL) {
        fprintf(stderr, "clCreateContext: %d\n", err);
        goto Leave;
    }

    // Create CommandQueue
    if ((queue = clCreateCommandQueue(context, device_id, 0, &err)) == NULL) {
        fprintf(stderr, "clCreateCommandQueue: %d\n", err);
        goto Leave;
    }

    // Create Program.
    if ((program = createProgram(context, device_id, "kernel/kernel.pz", &err)) == NULL) {
        fprintf(stderr, "clCreateProgram: %d\n", err);
        goto Leave;
    }

    // Create Kernel
    if ((kernel = clCreateKernel(program, "add", &err)) == NULL) {
        fprintf(stderr, "clCreateKernel: %d\n", err);
        goto Leave;
    }

    // Create Buffers
    if ((mem_src0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num, NULL, &err)) == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", err);
        goto Leave;
    }
    if ((mem_src1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num, NULL, &err)) == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", err);
        goto Leave;
    }
    if ((mem_dst = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * num, NULL, &err)) == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", err);
        goto Leave;
    }

    // Send source
    if ((err = clEnqueueWriteBuffer(queue, mem_src0, CL_TRUE, 0, sizeof(double) * num, src0, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer: %d\n", err);
        goto Leave;
    }
    if ((err = clEnqueueWriteBuffer(queue, mem_src1, CL_TRUE, 0, sizeof(double) * num, src1, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer: %d\n", err);
        goto Leave;
    }

    // Clear destination
    if ((err = clEnqueueFillBuffer(queue, mem_dst, &zero, sizeof(double), 0, sizeof(double) * num, 0, NULL, &write_event)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueFillBuffer: %d\n", err);
        goto Leave;
    }
    clWaitForEvents(1, &write_event);

    // Set kernel arguments
    if ((err = clSetKernelArg(kernel, 0, sizeof(size_t), &num)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        goto Leave;
    }
    if ((err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_dst)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        goto Leave;
    }
    if ((err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_src0)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        goto Leave;
    }
    if ((err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &mem_src1)) != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        goto Leave;
    }

    // Run device kernel.
    if ((err = executeKernel(device_id, queue, kernel)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", err);
        goto Leave;
    }

    // Get destination
    if ((err = clEnqueueReadBuffer(queue, mem_dst, CL_TRUE, 0, sizeof(double) * num, dst, 0, NULL, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer: %d\n", err);
        goto Leave;
    }

    // Finish all commands
    clFlush(queue);
    clFinish(queue);

Leave:
    if (write_event)
        clReleaseEvent(write_event);
    if (mem_src0)
        clReleaseMemObject(mem_src0);
    if (mem_src1)
        clReleaseMemObject(mem_src1);
    if (mem_dst)
        clReleaseMemObject(mem_dst);
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);
}

static bool verify(size_t num, const double* actual, const double* expected)
{
    bool   is_true     = true;
    size_t error_count = 0;

    for (size_t i = 0; i < num; ++i) {
        if (fabs(actual[i] - expected[i]) > 1.e-7) {
            if (error_count < 10) {
                fprintf(stderr, "# ERROR %zu %f %f\n", i, actual[i], expected[i]);
            }
            error_count++;
            is_true = false;
        }
    }

    return is_true;
}

int main(int argc, char* argv[])
{
    size_t num       = 1024;
    bool   succeeded = false;

    if (argc > 1) {
        num = strtol(argv[1], NULL, 10);
    }
    printf("Program  : %s\n", argv[0]);
    printf("Num      : %zu\n", num);

    double* src0    = calloc(num, sizeof(double));
    double* src1    = calloc(num, sizeof(double));
    double* dst_sc  = calloc(num, sizeof(double));
    double* dst_cpu = calloc(num, sizeof(double));

    if (src0 && src1 && dst_sc && dst_cpu) {
        initVector(num, src0);
        initVector(num, src1);
        cpuAdd(num, dst_cpu, src0, src1);
        pzcAdd(num, dst_sc, src0, src1);
        succeeded = verify(num, dst_sc, dst_cpu);
    } else {
        fprintf(stderr, "cannot allocate host memory.\n");
    }
    printf("%s\n", succeeded ? "PASS" : "FAIL");

    free(src0);
    free(src1);
    free(dst_sc);
    free(dst_cpu);

    return (succeeded ? 0 : 1);
}
