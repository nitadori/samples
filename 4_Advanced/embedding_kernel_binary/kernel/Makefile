PZSDK_PATH?=/opt/pzsdk.ver4.1
DEFAULT_MAKE=$(PZSDK_PATH)/make/default_pzcl_kernel.mk

PZC_TARGET_ARCH?=sc2

OBJ_DIR=obj.$(PZC_TARGET_ARCH)
TARGET=kernel.$(PZC_TARGET_ARCH).pz
PZCSRC=kernel.pzc

vpath %.pzc ../pzc

include $(DEFAULT_MAKE)

# Craete object file containing kernel binary
kernel.$(PZC_TARGET_ARCH).o: $(TARGET)
	objcopy -I binary -O elf64-x86-64 -B i386 \
		$< $@
