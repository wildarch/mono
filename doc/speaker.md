# Living room speaker
Documents how to set up the living room speaker

## Imaging the SD card
Insert a micro-SD card and run `bazel run //pimage` to create a new boot volume.
The password chosen in this step is only temporary, we will erase it later on when running the ansible playbook.

## Bootstrapping
Power on the pi and find its IP address.

Now set the hostname and reboot:

``` shell
ssh <ip>
sudo hostnamectl set-hostname speaker
sudo reboot
```

## Provisioning
Optionally verify the pi is reachable under hostname `speaker`:

```shell
ssh speaker.local
```

Run the playbook:

```shell
ansible-playbook -k ansible/speaker.yml
```