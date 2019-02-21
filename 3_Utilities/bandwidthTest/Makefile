PZSDK_PATH?=/opt/pzsdk.ver4.1
PZSDK_INC_PATH=$(PZSDK_PATH)/inc
PZSDK_LIB_PATH=$(PZSDK_PATH)/lib

CXX=g++
CXXFLAGS=-std=c++11 -Wall -g -O2

INCLUDE += -I$(PZSDK_INC_PATH)
LDFLAGS := -L$(PZSDK_LIB_PATH) -lpzcl -lpthread

TARGET=bandwidthTest

SRCS=main.cpp controller.cpp
OBJS=$(SRCS:.cpp=.o)
DEPS=$(SRCS:.cpp=.d)

.PHONY: $(TARGET) clean

all:$(TARGET)

$(TARGET):$(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -MMD $< -o $@

clean:
	-@rm $(TARGET) $(OBJS) $(DEPS)

run:
	./$(TARGET)

-include $(DEPS)
