#include "linux/export.h"
#include "linux/printk.h"
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");

const char EUDYPTULA_ID[] = "7c1caf2f50d1";
const size_t EUDYPTULA_ID_LEN = sizeof(EUDYPTULA_ID) - 1;

enum {
  CDEV_NOT_USED = 0,
  CDEV_EXCLUSIVE_OPEN = 1,
};

/* Is device open? Used to prevent multiple access to device */
static atomic_t already_open = ATOMIC_INIT(CDEV_NOT_USED);

static ssize_t
eudyptula_chardev_read(struct file *filp,   /* see include/linux/fs.h   */
                       char __user *buffer, /* buffer to fill with data */
                       size_t length,       /* length of the buffer     */
                       loff_t *offset) {
  loff_t bytes_left = ((loff_t)EUDYPTULA_ID_LEN) - *offset;
  if (bytes_left < 0) {
    return -EINVAL;
  } else if (bytes_left == 0) {
    // end of file
    *offset = 0;
    return 0;
  }

  size_t bytes_to_read = bytes_left;
  if (bytes_to_read > length) {
    bytes_to_read = length;
  }

  const char *start = EUDYPTULA_ID + *offset;
  for (size_t i = 0; i < bytes_to_read; i++) {
    put_user(start[i], &buffer[i]);
  }

  *offset += bytes_to_read;
  return bytes_to_read;
}

static ssize_t eudyptula_chardev_write(struct file *filp,
                                       const char __user *buff, size_t len,
                                       loff_t *off) {
  pr_alert("Sorry, this operation is not supported.\n");
  return -EINVAL;
}

static int eudyptula_chardev_open(struct inode *inode, struct file *file) {
  if (atomic_cmpxchg(&already_open, CDEV_NOT_USED, CDEV_EXCLUSIVE_OPEN))
    return -EBUSY;

  if (!try_module_get(THIS_MODULE)) {
    pr_err(
        "tried to open eudyptula device while the module is being removed\n");
    return -EBUSY;
  }

  return 0;
}

static int eudyptula_chardev_release(struct inode *inode, struct file *file) {
  // We're now ready for our next caller
  atomic_set(&already_open, CDEV_NOT_USED);

  // Decrement the usage count, or else once you opened the file, you will
  // never get rid of the module.
  module_put(THIS_MODULE);

  return 0;
}

static const struct file_operations eudyptula_chardev_fops = {
    .owner = THIS_MODULE,
    .read = eudyptula_chardev_read,
    .write = eudyptula_chardev_write,
    .open = eudyptula_chardev_open,
    .release = eudyptula_chardev_release,
};

static struct miscdevice eudyptula_miscdevice = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "eudyptula",
    .fops = &eudyptula_chardev_fops,
};

int init_module(void) {
  // Register the device
  int error = misc_register(&eudyptula_miscdevice);
  if (error) {
    pr_err("failed to register eudyptula device: %d\n", error);
    return error;
  }

  pr_info("loaded eudyptula character device\n");

  return 0;
}

void cleanup_module(void) {
  misc_deregister(&eudyptula_miscdevice);
  pr_info("unloaded eudyptula character device\n");
}