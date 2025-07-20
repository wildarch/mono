# Setting up NixOS on Raspberry Pi
I started from the official setup guide: https://nix.dev/tutorials/nixos/installing-nixos-on-a-raspberry-pi.
Also: https://nixos.wiki/wiki/NixOS_on_ARM/Raspberry_Pi_4.

The image can be downloaded from Hydra and directly written to the SD Card without unzipping using the Raspberry Pi imager tool.

## Testing Spotify Connect
```bash
nix-shell -p librespot
librespot
```

Failure

```bash
nixos-generate-config
# Modify it
sudo nixos-rebuild boot

# IMPORTANT: Set a password
passwd daan

sudo reboot
```

## Enable Audio
```bash
sudo mkdir /mnt/FIRMWARE
sudo mount /dev/disk/by-label/FIRMWARE /mnt/FIRMWARE/

# Add to /mnt/FIRMWARE/config.txt
dtparam=audio=on
hdmi_ignore_edid_audio=1

sudo nixos-rebuild boot
sudo reboot
```

## Final nix configuration
See configuration file in this folder.
