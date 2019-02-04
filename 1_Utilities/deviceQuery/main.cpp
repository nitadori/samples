/**
 * @copyright 2018 PEZY Computing, K.K.
 */

#include "CL/cl.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
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
  } else {
    std::clog << devs.size() << " device(s) found" << std::endl;
  }

  // Query device info
  for (size_t i = 0; i < devs.size(); ++i) {
    std::clog << "-------------------------------------------" << std::endl;
    std::clog << "Device ID    : " << i << std::endl;
    cl::Device dev = devs[i];
    std::clog << "Name         : " << dev.getInfo<CL_DEVICE_NAME>()
              << std::endl;
    std::clog << "Vender       : " << dev.getInfo<CL_DEVICE_VENDOR>()
              << std::endl;
    std::clog << "Max Work Item: "
              << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << std::endl;
  }
  std::clog << "-------------------------------------------" << std::endl;

  return 0;
}
