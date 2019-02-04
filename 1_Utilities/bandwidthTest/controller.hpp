#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <functional>

namespace pezy {
	enum MEMMODE
	{
		PAGEABLE = 0,
		PINNED
	};

	enum MODE
	{
		QUICK = 0,
		RANGE
	};

	enum MEASURE
	{
		HtoD = 0,
		DtoH,
		ALL
	};

	typedef struct {
		size_t device_id;
		MEMMODE mem_mode;
		MODE mode;
		MEASURE measure;
		size_t range_start;
		size_t range_end;
		size_t range_inc;
	} param_t;

	class Controller
	{
	public:
		Controller(size_t id_);
		void runTest(const param_t& params);

	private:
		size_t device_id;
		size_t loop_count;
		cl::Context context;
		cl::CommandQueue queue;

		void init();
		void showDeviceInfo() const;

		void test(MEMMODE mem_mode, MEASURE measure, size_t range_start, size_t range_end, size_t range_inc);
		void testOneShot(void* ptr, cl::Buffer& buf, size_t size, std::function<void(cl::Buffer&, size_t, void*)> trans);
	};
}

#endif
