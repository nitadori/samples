PZSDK_PATH?=/opt/pzsdk.ver4.1
DEFAULT_MAKE=$(PZSDK_PATH)/make/default_pzcl_host.mk

TARGET=Atomic
CPPSRC=main.cpp
CCOPT=-O2 -Wall -D__LINUX__ -DNDEBUG -std=c++11 -fopenmp
LDOPT=-fopenmp

INC_DIR?=

LIB_DIR?=

PZCL_KERNEL_DIRS=kernel

# supported archtecture is sc2 or later. (sc1/sc1-64 does not support atomic functions)
PZC_TARGET_ARCH?=sc2
export PZC_TARGET_ARCH

include $(DEFAULT_MAKE)

run:
	@./$(TARGET) 102400
