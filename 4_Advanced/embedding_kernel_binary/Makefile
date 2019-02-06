# path to PZSDK
PZSDK_PATH?=/opt/pzsdk.ver4.1
export PZSDK_PATH

# kernel build directory
PZCL_KERNEL_DIRS = kernel.sc1-64 kernel.sc2

# object files containing kernel binary
PZCL_KERNEL_OBJS = $(addsuffix /kernel.o, $(PZCL_KERNEL_DIRS))

CXX      = c++
CXXFLAGS = -O2 -std=c++11 -Wall -Wextra -Wcast-align -Wcast-qual -I $(PZSDK_PATH)/inc

LD      = c++
LDFLAGS = -lm -lpthread -ldl -lrt -L $(PZSDK_PATH)/lib -lpzcl

PROG    = pzcAdd
SRCS    = main.cpp
OBJS    = $(addsuffix .o, $(basename $(SRCS)))

all: host kernel

$(PROG): $(OBJS) $(PZCL_KERNEL_OBJS)
	$(LD) -o $(PROG) $(OBJS) $(PZCL_KERNEL_OBJS) $(LDFLAGS)

$(PZCL_KERNEL_OBJS): kernel

host: $(PROG)

kernel:
	for dirs in $(PZCL_KERNEL_DIRS); do $(MAKE) -C $$dirs kernel.o; done

run: host kernel
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PZSDK_PATH)/lib ./$(PROG) 102400

clean:
	rm -f $(PROG)
	rm -f $(OBJS)
	rm -f $(PZCL_KERNEL_OBJS)
	for dirs in $(PZCL_KERNEL_DIRS); do $(MAKE) -C $$dirs clean; done


.PHONY: all host kernel run clean
