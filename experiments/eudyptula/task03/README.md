# Task 3
Now that you have your custom kernel up and running, it's time to modify
it!

The tasks for this round is:
  - take the kernel git tree from Task 02 and modify the Makefile to
    and modify the EXTRAVERSION field.  Do this in a way that the
    running kernel (after modifying the Makefile, rebuilding, and
    rebooting) has the characters "-eudyptula" in the version string.
  - show proof of booting this kernel.  Extra cookies for you by
    providing creative examples, especially if done in intrepretive
    dance at your local pub.
  - Send a patch that shows the Makefile modified.  Do this in a manner
    that would be acceptable for merging in the kernel source tree.
    (Hint, read the file Documentation/SubmittingPatches and follow the
    steps there.)

Remember to use your ID assigned to you in the Subject: line when
responding to this task, so that I can figure out who to attribute it
to.

## Building the new kernel
```bash
make -j`nproc` bindeb-pkg
```

## Installing it
```bash
cd /home/daan/workspace/mono/experiments/eudyptula
sudo apt install ./linux-image-6.6.0-eudyptula-g1acfd2bd3f0d-dirty_6.6.0-eudyptula-g1acfd2bd3f0d-6_amd64.deb
sudo reboot
```

After the reboot, the VM should report that the new kernel is used:
```bash
daan@daan-zbook:~$ multipass shell eudyptula 

Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 6.6.0-eudyptula-g1acfd2bd3f0d-dirty x86_64)
...
```

Format for the patch: https://www.kernel.org/doc/html/v4.19/process/submitting-patches.html#the-canonical-patch-format