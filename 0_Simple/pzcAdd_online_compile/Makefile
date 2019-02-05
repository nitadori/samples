PZSDK_PATH?=/opt/pzsdk.ver4.1
DEFAULT_MAKE=$(PZSDK_PATH)/make/default_pzcl_host.mk

TARGET=pzcAdd_online_compile
CPPSRC=main.cpp
CCOPT=-O2 -Wall -D__LINUX__ -DNDEBUG -std=c++11

INC_DIR?=

LIB_DIR?=

PZCL_KERNEL_DIRS=

# supported archtecture:
# sc1-64, sc2
#
# limitation:
# sc1 has a limitation. Please refer to the release note in the pzsdk.

include $(DEFAULT_MAKE)

run:
	@./$(TARGET) 102400
