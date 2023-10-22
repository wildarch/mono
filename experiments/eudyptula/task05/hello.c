#include "linux/usb.h"
#include <linux/hid.h>
#include <linux/mod_devicetable.h>
#include <linux/module.h>
#include <linux/usb/input.h>

static struct usb_device_id hello_table[] = {
    // Stolen from drivers/hid/usbhid/usbkbd.c
    {USB_INTERFACE_INFO(USB_INTERFACE_CLASS_HID, USB_INTERFACE_SUBCLASS_BOOT,
                        USB_INTERFACE_PROTOCOL_KEYBOARD)},
    {} /* Terminating entry */
};

int hello_probe(struct usb_interface *intf, const struct usb_device_id *id) {
  pr_debug("Hello USB!");
  // Not taking ownership of this device.
  return -ENODEV;
}

void hello_disconnect(struct usb_interface *intf) {
  // Nothing
}

static struct usb_driver hello_driver = {
    .name = "hello",
    .probe = hello_probe,
    .disconnect = hello_disconnect,
    .id_table = hello_table,
};

module_usb_driver(hello_driver);

MODULE_DEVICE_TABLE(usb, hello_table);

MODULE_LICENSE("GPL");