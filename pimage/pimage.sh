#!/bin/bash
set -e

echo "=== Raspberry Pi semi-automated imager ==="
echo

if [ -z "$IMAGE" ]; then
    echo "Error: env variable IMAGE must be given"
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "Error: image $IMAGE does not exist"
    exit 1
fi

echo "Image to flash: $IMAGE"

echo "=== Select Disk ==="
# Exclude devices with major node type 7 (loop devices)
lsblk -e7

read -rp "Enter disk to use [sda]: " DISK
DISK="${DISK:-sda}"
echo "Will use disk $DISK"

DEVICE="/dev/$DISK"

if lsblk "$DEVICE" | grep -q "$DISK"; then
    lsblk "$DEVICE"
else
    echo "Error: disk $DISK not found"
    exit 1
fi

# Unmount any partitions if necessary
grep "^$DEVICE" /proc/mounts | awk '{print $1}' | while read -r part; do
    echo "$part needs to be unmounted"
    sudo umount "$part"
done

echo "=== Write Image ==="

DD_COMMAND="xz -d < $IMAGE - | dd iflag=fullblock of=$DEVICE oflag=direct bs=1M status=progress conv=fsync"

echo "Please confirm the below commmand is correct:"
echo "$DD_COMMAND"
echo "WARNING: This will wipe the contents of $DISK entirely"
read -rp "Do you want to continue? (y/n/skip) " OK

if [ "$OK" = "y" ]; then
    sudo sh -c "$DD_COMMAND"
elif [ "$OK" = "skip" ]; then
    echo "Skipping image write"
else
    echo "Cancelled"
    exit 0
fi

echo "=== Boot Configuration ==="

BOOT_DEV="${DEVICE}1"
if [ ! -e "$BOOT_DEV" ]; then
    echo "Boot partition $BOOT_DEV not found (try removing and inserting the disk)"
    exit 1
fi

mkdir -p /tmp/pimage_boot
sudo mount "$BOOT_DEV" /tmp/pimage_boot

echo "SSH service enabled"
sudo touch /tmp/pimage_boot/SSH

# Setup WiFi
SSID="Matt's warehouse"
read -rp "Enter WiFi password for '$SSID': " WIFI_PASSWORD
echo "Wrote WiFi settings:"
sudo tee /tmp/pimage_boot/wpa_supplicant.conf <<EOT
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
country=NL
update_config=1

network={
 ssid="$SSID"
 psk="$WIFI_PASSWORD"
}
EOT

# Configure a user
read -rp "Enter login password for $USER: " LOGIN_PASSWORD
LOGIN_PASSWORD_ENCRYPTED=$(echo "$LOGIN_PASSWORD" | $OPENSSL passwd -6 -stdin)
echo "Wrote user configuration:"
sudo tee /tmp/pimage_boot/userconf.txt <<EOT
$USER:$LOGIN_PASSWORD_ENCRYPTED
EOT

sudo umount /tmp/pimage_boot
rm -r /tmp/pimage_boot