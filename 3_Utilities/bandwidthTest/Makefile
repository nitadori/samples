PZSDK_PATH  ?= /opt/pzsdk.ver4.1
DEFAULT_MAKE = $(PZSDK_PATH)/make/default_pzcl_host.mk

CCOPT = -std=c++11 -Wall -g -O2

TARGET = bandwidthTest
CPPSRC = main.cpp controller.cpp

include $(DEFAULT_MAKE)

run:
	@./$(TARGET)
