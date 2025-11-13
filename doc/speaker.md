# Living room speaker
Documents how to set up the living room speaker.

## Imaging the SD Card
Use `rpi-imager`. It is installed from the regular repositories:

```bash
sudo apt install rpi-imager
```

Set hostname `speaker`, username `daan`, and enable the SSH server.

## Provisioning
Optionally verify the pi is reachable under hostname `speaker`:

```shell
ssh speaker.local
```

Run the playbook:

```shell
ansible-playbook ansible/speaker.yml
```

## Last Update (2025-11-13)
Use `rpi-imager`. It is installed from the regular repositories:

```bash
sudo apt install rpi-imager
```

Set hostname `speaker`, username `daan`, and enable the SSH server.
I added to `/boot/firmware/config.txt`:
```
dtoverlay=disable-wifi
```

I setup a Spotify Connect server using [raspotify](https://github.com/dtcooper/raspotify):

```bash
sudo apt-get -y install curl && curl -sL https://dtcooper.github.io/raspotify/install.sh | sh
```

To `/etc/raspotify/conf` I add:

```bash
LIBRESPOT_NAME="Songs for the elderly"
```