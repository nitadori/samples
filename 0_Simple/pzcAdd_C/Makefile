# path to PZSDK
PZSDK_PATH?=/opt/pzsdk.ver4.1
export PZSDK_PATH

# supported archtecture:
# sc1-64, sc2
PZC_TARGET_ARCH?=sc2
export PZC_TARGET_ARCH

# kernel build directory
PZCL_KERNEL_DIR = kernel

CC      = cc
CFLAGS  = -O2 -std=c99 -Wall -Wextra -Wcast-align -Wcast-qual -I $(PZSDK_PATH)/inc

# Use c++ instead of cc, because libpzcl requires C++ runtimes
LD      = c++
LDFLAGS = -lm -lpthread -ldl -lrt -L $(PZSDK_PATH)/lib -lpzcl

PROG    = pzcAdd
SRCS    = main.c
OBJS    = $(addsuffix .o, $(basename $(SRCS)))


all: host kernel

$(PROG): $(OBJS)
	$(LD) -o $(PROG) $(OBJS) $(LDFLAGS)

host: $(PROG)

kernel:
	$(MAKE) -C $(PZCL_KERNEL_DIR)

run: host kernel
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PZSDK_PATH)/lib ./$(PROG) 102400

clean:
	rm -f $(PROG)
	rm -f $(OBJS)
	$(MAKE) -C $(PZCL_KERNEL_DIR) clean


.PHONY: all host kernel run clean
