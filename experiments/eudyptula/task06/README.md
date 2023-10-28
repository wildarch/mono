# Task 6
Nice job with the module loading macros, those are tricky, but a very
valuable skill to know about, especially when running across them in
real kernel code.

Speaking of real kernel code, let's write some!

The task this time is this:
  - Take the kernel module you wrote for task 01, and modify it to be a
    misc char device driver.  The misc interface is a very simple way to
    be able to create a character device, without having to worry about
    all of the sysfs and character device registration mess.  And what a
    mess it is, so stick to the simple interfaces wherever possible.
  - The misc device should be created with a dynamic minor number, no
    need running off and trying to reserve a real minor number for your
    test module, that would be crazy.
  - The misc device should implement the read and write functions.
  - The misc device node should show up in /dev/eudyptula.
  - When the character device node is read from, your assigned id is
    returned to the caller.
  - When the character device node is written to, the data sent to the
    kernel needs to be checked.  If it matches your assigned id, then
    return a correct write return value.  If the value does not match
    your assigned id, return the "invalid value" error value.
  - The misc device should be registered when your module is loaded, and
    unregistered when it is unloaded.
  - Provide some "proof" this all works properly.

## Getting `compile_commands.json`

```bash
bear -- make KERNEL=/home/daan/workspace/mono/experiments/eudyptula/linux CC=clang-17
```

## Loading
```bash
# This is where our module is
cd /home/daan/workspace/mono/experiments/eudyptula/task06

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