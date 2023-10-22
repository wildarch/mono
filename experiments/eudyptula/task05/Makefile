obj-m += hello.o 
 
PWD := $(CURDIR) 
 
all: 
	make -C ../linux M=$(PWD) modules 
 
clean: 
	make -C ../linux M=$(PWD) clean