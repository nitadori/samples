/*!
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 */

#include "pezy.hpp"
#include <cstring>
#include <iostream>
#include <string>

#include <getopt.h>
#include <stdexcept>

namespace {
void PrintMessages(size_t STREAM_ARRAY_SIZE, size_t NTIMES, size_t OFFSET, size_t SizeOfStream)
{
    const std::string HLINE = "-------------------------------------------------------------";

    std::cout << HLINE << std::endl;
    std::cout << "STREAM version $Revision : 5.10 $" << std::endl;
    std::cout << HLINE << std::endl;

    std::cout << "This system uses " << SizeOfStream << " bytes per array element." << std::endl;
    std::cout << HLINE << std::endl;

    std::cout << "Array Size = " << STREAM_ARRAY_SIZE << " (elements), Offset = " << OFFSET << " (elements) " << std::endl;
    std::cout << "Memory per array = " << SizeOfStream * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0) << " MiB (= " << SizeOfStream * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0) << " GiB)." << std::endl;
    std::cout << "Total Memory required = " << (3.0 * SizeOfStream) * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0) << " MiB (= " << (3.0 * SizeOfStream) * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0) << " GiB). " << std::endl;

    std::cout << "Each kernel will be executed " << NTIMES << " times." << std::endl;
}

void ShowSummary(const std::vector<std::vector<double>>& times, size_t STREAM_ARRAY_SIZE, size_t NTIMES)
{
    using STREAM_TYPE = double;

    // Summary
    std::vector<double> avgtime(4, 0);
    std::vector<double> mintime(4, std::numeric_limits<double>::max());
    std::vector<double> maxtime(4, std::numeric_limits<double>::min());

    for (auto k = decltype(NTIMES)(1); k < NTIMES; ++k) {
        const auto& cur_time = times[k];

        for (size_t j = 0; j < 4; ++j) {
            avgtime[j] += cur_time[j];
            mintime[j] = std::min(mintime[j], cur_time[j]);
            maxtime[j] = std::max(maxtime[j], cur_time[j]);
        }
    }

    const std::string label[4] = { "Copy:", "Scale:",
                                   "Add:", "Triad:" };

    const double bytes[4] = { (double)2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
                              (double)2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
                              (double)3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
                              (double)3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE };

    printf("Function\tBest Rate MB/s \tAvg time\tMin time\tMax time\n");
    for (size_t j = 0; j < 4; ++j) {
        auto avg = avgtime[j] / static_cast<double>(NTIMES - 1);

        printf("%s\t\t%12.1f\t%11.6f\t%11.6f\t%11.6f\n", label[j].c_str(),
               1.0e-6 * bytes[j] / mintime[j],
               avg, mintime[j], maxtime[j]);
    }
    const std::string HLINE = "-------------------------------------------------------------";
    std::cout << HLINE << std::endl;
}
size_t stream_array_size = static_cast<size_t>(100000000);
size_t ntimes            = 10;
size_t offset            = 0;
size_t device_id         = 0;

void usage(const std::string& bin_name)
{
    std::cout << bin_name << " [OPTIONS] " << std::endl;
    std::cout << "Test bandwidth device memory." << std::endl;
    std::cout << "\n";

    std::cout << "Options: " << std::endl;
    std::cout << "-h, --help\tDisplay this help menu." << std::endl;
    std::cout << "-d [device no], --device=[device no]\tSpecify device No to be used.\n"
              << "   [device no] = 0,1,2,...n\t\tSpecify any particular device to be used." << std::endl;
    std::cout << "-s [array size], --size=[array size]\tSpecify array size to use." << std::endl;
}

int parseArgs(int argc, char** argv)
{
    const char*         optstring  = "hd:s";
    const struct option longopts[] = {
        //{    *name,           has_arg, *flag, val },
        { "help", no_argument, nullptr, 'h' },
        { "device", required_argument, nullptr, 'd' },
        { "size", required_argument, nullptr, 's' }
    };

    int c;
    int longindex;

    auto atoiKMGT = [](const char* str) {
        int  c = str[strlen(str) - 1];
        long n = atoi(str);
        switch (c) {
        case 'T':
        case 't':
            n *= 1024;
        case 'G':
        case 'g':
            n *= 1024;
        case 'M':
        case 'm':
            n *= 1024;
        case 'K':
        case 'k':
            n *= 1024;
        }

        return n;
    };

    while ((c = getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
        if (c == 'h') {
            usage(argv[0]);
            return -1;
        } else if (c == 'd') {
            device_id = strtol(optarg, nullptr, 10);
        } else if (c == 's') {
            stream_array_size = atoiKMGT(optarg);
        } else {
            // invalid option
            return -1000;
        }
    }

    return 0;
}
};

int main(int argc, char** argv)
{
    if (parseArgs(argc, argv) < 0) {
        return -1;
    }

    PrintMessages(stream_array_size, ntimes, offset, sizeof(double));

    try {
        pezy handler(device_id);
        auto times = handler.run(stream_array_size, ntimes, offset);
        ShowSummary(times, stream_array_size, ntimes);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
