# Task 2
Now that you have written your first kernel module, it's time to take
off the training wheels and move on to building a custom kernel.  No
more distro kernels for you, for this task you must run your own kernel.
And use git!  Exciting isn't it!  No, oh, ok...

The tasks for this round is:
  - download Linus's latest git tree from git.kernel.org (you have to
    figure out which one is his, it's not that hard, just remember what
    his last name is and you should be fine.)
  - build it, install it, and boot it.  You can use whatever kernel
    configuration options you wish to use, but you must enable
    CONFIG_LOCALVERSION_AUTO=y.
  - show proof of booting this kernel.  Bonus points for you if you do
    it on a "real" machine, and not a virtual machine (virtual machines
    are acceptable, but come on, real kernel developers don't mess
    around with virtual machines, they are too slow.  Oh yeah, we aren't
    real kernel developers just yet.  Well, I'm not anyway, I'm just a
    script...)  Again, proof of running this kernel is up to you, I'm
    sure you can do well.

Hint, you should look into the 'make localmodconfig' option, and base
your kernel configuration on a working distro kernel configuration.
Don't sit there and answer all 1625 different kernel configuration
options by hand, even I, a foolish script, know better than to do that!

After doing this, don't throw away that kernel and git tree and
configuration file.  You'll be using it for later tasks, a working
kernel configuration file is a precious thing, all kernel developers
have one they have grown and tended to over the years.  This is the
start of a long journey with yours, don't discard it like was a broken
umbrella, it deserves better than that.

Remember to use your ID assigned to you in the Subject: line when
responding to this task, so that I can figure out who to attribute it
to.

## Getting the kernel
I am placing it in the main `eudyptula` folder because it will be used in multiple tasks.

```bash
git clone git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git

# The specific commit we built, for future reference.
git rev-parse HEAD
# Output: 1acfd2bd3f0d9dc34ea1871a445c554220945d9f

cd linux/

make localmodconfig
make olddefconfig

make -j`nproc` bindeb-pkg LOCALVERSION=-eudyptula
```

In the VM:

```bash
cd /home/daan/workspace/mono/experiments/eudyptula
sudo apt install ./linux-image-6.6.0-rc6-eupdyptula-00334-g1acfd2bd3f0d_6.6.0-rc6-00334-g1acfd2bd3f0d-5_amd64.deb
sudo reboot
```

After the reboot, the VM should report that the new kernel is used:
```bash
daan@daan-zbook:~$ multipass shell eudyptula 

Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 6.6.0-rc6-eupdyptula-00334-g1acfd2bd3f0d x86_64)
...
```