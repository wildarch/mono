# Headscale setup
Headscale runs on the server (oracle).

## Users
There should already be two users, one per connected device.
If you need to recreate them:

```bash
sudo headscale users create thinkpad
sudo headscale users create zbook
```

## Register Node
To connect a device after a reinstall, do the following.

```bash
# NOTE: http:// protocol is required
sudo tailscale up --login-server http://headscale.wildarch.dev
```

A browser window with further instructions is opened and contains the value for `<YOUR_MACHINE_KEY>`. Approve and register the node on the headscale server:

```bash
sudo headscale nodes register --user <USER> --key <YOUR_MACHINE_KEY>
```

Check that you are successfully connected:

```bash
tailscale status
```