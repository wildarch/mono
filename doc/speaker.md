# Living room speaker
Documents how to set up the living room speaker.

## Imaging the SD Card
Use `rpi-imager`. It is installed from the regular repositories:

```bash
sudo apt install rpi-imager
```

Set hostname `speaker`, username `daan`, copy the SSH key and configure the Wi-Fi credentials.
The automatic Wi-Fi configuration did not work for me.
I had to connect a screen and keyboard and configure it using `raspi-config` after first boot.

## Provisioning
Optionally verify the pi is reachable under hostname `speaker`:

```shell
ssh speaker.local
```

Run the playbook:

```shell
ansible-playbook ansible/speaker.yml
```