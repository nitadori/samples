#include <iostream>
#include <string>

#include "controller.hpp"
#include <getopt.h>

#include <stdexcept>

namespace {
	void usage(const std::string& bin_name)
	{
		std::cout << bin_name << " [OPTIONS] " << std::endl;
		std::cout << "Test bandwidth for host to device, device to host transfers" << std::endl;
		std::cout << "\n";

		std::cout << "Options: " << std::endl;
		std::cout << "-h, --help\tDisplay this help menu" << std::endl;
		std::cout << "--device=[device no]\tSpecify device no to be used\n"
				  << "  0,1,2,...n - Specify any particular device to be used" << std::endl;
		std::cout << "--memory=[MEM MODE]\tSpecify which memory mode to use\n"
				  << "  pageable - pageable memory\n"
				  << "  pinned   - non-pageable system memory" << std::endl;
		std::cout << "--mode=[MODE]\tSpecify the mode to use\n"
				  << "  quick - performs a quick measurement\n"
				  << "  range - measures a user-specified range of values" << std::endl;
		std::cout << "--htod\tMeasure host to device transfers" << std::endl;
		std::cout << "--dtoh\tMeasure device to host transfers" << std::endl;

		std::cout << "\n";

		std::cout << "Range mode options" << std::endl;
		std::cout << "--start=[SIZE]\tStarting transfer size in bytes" << std::endl;
		std::cout << "--end=[SIZE]\tEnding transfer size in bytes" << std::endl;
		std::cout << "--increment=[SIZE]\tIncrement size in bytes" << std::endl;
	}

	int parseArgs(int argc, char** argv, pezy::param_t& params)
	{
		const char* optstring = "h";
		const struct option longopts[] = {
			//{    *name,           has_arg, *flag, val },
			{"help", no_argument, nullptr, 'h'},
			{"device", required_argument, nullptr, 'd'},
			{"memory", required_argument, nullptr, 'p'},
			{"mode", required_argument, nullptr, 'm'},
			{"htod", no_argument, reinterpret_cast<int*>(&params.measure), pezy::HtoD},
			{"dtoh", no_argument, reinterpret_cast<int*>(&params.measure), pezy::DtoH},
			{"start", required_argument, nullptr, 's'},
			{"end", required_argument, nullptr, 'e'},
			{"increment", required_argument, nullptr, 'i'}
		};

		// set default mode
		params.device_id = 0;
		params.mem_mode = pezy::PINNED;
		params.mode = pezy::QUICK;
		params.measure = pezy::ALL;
		params.range_start = 1024; // 1KB
		params.range_end = 64 * 1024 * 1024;  // 64MB
		params.range_inc = 1024; // 1KB

		int c;
		int longindex;

		auto atoiKMGT = [](const char* str) {
			int c = str[strlen(str) - 1];
			long n = atoi(str);
			switch(c) {
			case 'T':
			case 't': n *= 1024;
			case 'G':
			case 'g': n *= 1024;
			case 'M':
			case 'm': n *= 1024;
			case 'K':
			case 'k': n *= 1024;
			}

			return n;
		};

		while((c=getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
			if(c == 0) {
				// nop
			} else if(c == 'h') {
				usage(argv[0]);
				return -1;
			} else if(c == 'd') {
				//std::cout << "device " << optarg << std::endl;
				size_t device_id = strtol(optarg, nullptr, 10);
				params.device_id = device_id;
			} else if(c == 'p') {
				std::string mem_mode = optarg;
				std::cout << "mem_mode " << mem_mode << std::endl;
				if(mem_mode == "pageable") {
					params.mem_mode = pezy::PAGEABLE;
				} else if(mem_mode == "pinned") {
					params.mem_mode = pezy::PINNED;
				} else {
					std::cerr << "Invalid memory mode. valid modes are pageable or pinned.\n"
							  << "See --help for more information." << std::endl;
					return -1000;
				}
			} else if(c == 'm') {
				std::string mode = optarg;
				std::cout << "mode " << mode << std::endl;
				if(mode == "quick") {
					params.mode = pezy::QUICK;
				} else if(mode == "range") {
					params.mode = pezy::RANGE;
				} else {
					std::cerr << "Invalid mode. valid modes are quick or range.\n"
							  << "See --help for more information." << std::endl;
					return -2000;
				}
			} else if(c == 's') {
				size_t start = atoiKMGT(optarg);
				params.range_start = start;
			} else if(c == 'e') {
				size_t end = atoiKMGT(optarg);
				params.range_end = end;
			} else if(c == 'i') {
				size_t inc = atoiKMGT(optarg);
				params.range_inc = inc;
			}
		}

		if(params.mode == pezy::RANGE) {
			if(params.range_start > params.range_end) {
				std::cout << "Invalid start and end value." << std::endl;
				return -3000;
			} else if(params.range_start < 0){
				std::cout << "Invalid range start value." << std::endl;
				return -4000;
			} else if(params.range_end > ((size_t)32 * 1024 * 1024 * 1024)) { // over 32GB
				std::cout << "Invalid range end value." << std::endl;
				return -5000;
			}
		}

		return 0;
	}
}

int main(int argc, char** argv)
{
	// usage(argv[0]);

	pezy::param_t params;
	int ret = parseArgs(argc, argv, params);
	if(ret != 0) {
		return -1;
	}

	try {
		pezy::Controller controller(params.device_id);

		controller.runTest(params);

	} catch(const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
