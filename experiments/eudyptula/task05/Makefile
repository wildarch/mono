obj-m += hello.o 

KERNEL_SRC := /lib/modules/$(shell uname -r)/build
 
PWD := $(CURDIR) 
 
all: 
	make -C $(KERNEL_SRC) M=$(PWD) modules 
 
clean: 
	make -C $(KERNEL_SRC) M=$(PWD) clean