PZSDK_PATH?=/opt/pzsdk.ver4.1
DEFAULT_MAKE=$(PZSDK_PATH)/make/default_pzcl_kernel.mk

PZC_TARGET_ARCH=sc2

TARGET=kernel.pz
PZCSRC=kernel.pzc

vpath %.pzc ../pzc

include $(DEFAULT_MAKE)

# Craete object file containing kernel binary
kernel.o: $(TARGET)
	objcopy -I binary -O elf64-x86-64 -B i386 \
		--prefix-symbols=$(subst -,_,$(PZC_TARGET_ARCH)) \
		$< $@
