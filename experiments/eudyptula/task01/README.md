# Task 1
Write a Linux kernel module, and stand-alone Makefile, that when loaded
prints to the kernel debug log level, "Hello World!"  Be sure to make
the module be able to be unloaded as well.

The Makefile should build the kernel module against the source for the
currently running kernel, or, use an environment variable to specify
what kernel tree to build it against.

Please show proof of this module being built, and running, in your
kernel.  What this proof is is up to you, I'm sure you can come up with
something.  Also be sure to send the kernel module you wrote, along with
the Makefile you created to build the module.

Remember to use your ID assigned to you in the Subject: line when
responding to this task, so that I can figure out who to attribute it
to.  You can just respond to the task with the answers and all should be
fine.

If you forgot, your id is "7c1caf2f50d1".  But of course you have not
forgotten that yet, you are better than that.

## Loading
```bash
# This is where our module is
cd /home/daan/workspace/mono/experiments/eudyptula/task01

# Print module info
modinfo hello.ko

# Should find nothing
sudo lsmod | grep hello

# Load it
sudo insmod hello.ko

# Should return the loaded module
sudo lsmod | grep hello

# Unload it
sudo rmmod hello

# Check the logs
sudo journalctl --since "1 hour ago" | grep kernel
```

The logs should contain something like:

```
Oct 22 15:19:43 eudyptula kernel: Hello World!
```

You may also see messages about the kernel being tainted, which is not a problem:

```
Oct 22 15:18:39 eudyptula kernel: hello: loading out-of-tree module taints kernel.
Oct 22 15:18:39 eudyptula kernel: hello: module verification failed: signature and/or required key missing - tainting kernel
Oct 22 15:19:43 eudyptula kernel: Hello World!
```