# path to PZSDK
PZSDK_PATH?=/opt/pzsdk.ver4.1
export PZSDK_PATH

# kernel architectures
# supported architecture:
# sc1-64, sc2
PZC_ARCHITECTURES = sc1-64 sc2

# object files containing kernel binary
PZCL_KERNEL_OBJS = $(addsuffix .o, $(addprefix kernel/kernel., $(PZC_ARCHITECTURES)))

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
	for arch in $(PZC_ARCHITECTURES); do \
		$(MAKE) -C kernel PZC_TARGET_ARCH=$$arch kernel.$$arch.o; \
	done

run: host kernel
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(PZSDK_PATH)/lib ./$(PROG) 102400

clean:
	rm -f $(PROG)
	rm -f $(OBJS)
	rm -f $(PZCL_KERNEL_OBJS)
	for arch in $(PZC_ARCHITECTURES); do \
		$(MAKE) -C kernel PZC_TARGET_ARCH=$$arch clean; \
	done

.PHONY: all host kernel run clean
