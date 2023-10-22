# Eudyptula Challenges
My attempt at [The Eudyptula Challenge](http://eudyptula-challenge.org/).

While the program is not accepting new applications at the time of writing, I did find [a description of the tasks with solutions](https://github.com/agelastic/eudyptula).

## Development Setup
Loading untested kernel modules into your running kernel can have disastrous effects.
To avoid my laptop crashly horribly, I will develop and test the kernel modules in a virtual machine.

I run Ubuntu, so [multipass](https://ubuntu.com/server/docs/virtualization-multipass) seems like a good choice.
Installing it and setting up a basic VM is very easy.

### Setting up the VM
Get the host kernel version:

```bash
uname -r
# 5.19.0-50-generic
```

Start the VM:

```bash
# Matches my host OS
multipass launch 22.04 --name eudyptula 

multipass shell eudyptula
```

Set it up:
```bash
sudo apt update
sudo apt upgrade
sudo apt install linux-image-unsigned-5.19.0-50-generic
```

### Mounting this directory to the VM
```bash
multipass mount /home/daan/workspace/mono/experiments/eudyptula/ eudyptula
```

## Resources
- [The Linux Kernel Module Programming Guide](https://sysprog21.github.io/lkmpg/)
- [Working with the kernel development community](https://www.kernel.org/doc/html/v4.19/process/index.html)