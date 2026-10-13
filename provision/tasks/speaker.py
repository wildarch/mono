"""Raspberry Pi living-room speaker tasks (mirrors ``ansible/speaker.yml``)."""

import os

from framework import register_task

SPOTIFYD_VERSION = "v0.3.5"
SPOTIFYD_URL = (
    "https://github.com/Spotifyd/spotifyd/releases/download/"
    f"{SPOTIFYD_VERSION}/spotifyd-linux-armv6-slim.tar.gz"
)

SPOTIFYD_SERVICE = """\
[Unit]
Description=A spotify playing daemon
Documentation=https://github.com/Spotifyd/spotifyd
Wants=sound.target
After=sound.target
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=/usr/bin/spotifyd --no-daemon --device-name="Songs for the Elderly" --device-type=speaker
Restart=always
RestartSec=12

[Install]
WantedBy=default.target
"""

ASOUND_CONF = """\
pcm.!default {
  type hw card 0
}
ctl.!default {
  type hw card 0
}
"""


def speaker_authorized_keys(ctx):
    """Install the user's SSH public keys for passwordless login."""
    ssh_dir = os.path.join(ctx.home, ".ssh")
    authorized = os.path.join(ssh_dir, "authorized_keys")
    keys = []
    for name in ("id_rsa.pub", "id_ed25519.pub"):
        path = os.path.join(ssh_dir, name)
        if os.path.exists(path):
            with open(path) as fh:
                keys.append(fh.read().strip())

    if not keys:
        return False

    existing = ""
    if os.path.exists(authorized):
        with open(authorized) as fh:
            existing = fh.read()

    lines = existing.splitlines()
    changed = False
    for key in keys:
        if key not in lines:
            lines.append(key)
            changed = True
    if changed:
        ctx.write_file(
            authorized,
            "\n".join(lines) + "\n",
            mode=0o600,
            owner=ctx.user,
            group=ctx.user,
        )
    return changed


register_task("speaker_authorized_keys", speaker_authorized_keys, root=True)


def speaker_audio_config(ctx):
    """Configure the HiFiBerry DAC and disable onboard/HDMI audio."""
    changed = False
    config = "/boot/config.txt"
    if os.path.exists(config):
        changed |= ctx.line_in_file(config, regexp=r"dtparam=audio=on", line=None)
        changed |= ctx.line_in_file(
            config,
            regexp=r"^dtoverlay=vc4-kms-v3d",
            line="dtoverlay=vc4-kms-v3d,audio=off",
        )
        changed |= ctx.line_in_file(
            config,
            regexp=r"^dtoverlay=hifiberry-dac",
            line="dtoverlay=hifiberry-dac",
        )

    changed |= ctx.write_file("/etc/asound.conf", ASOUND_CONF, mode=0o644)
    return changed


register_task("speaker_audio_config", speaker_audio_config, root=True)


def speaker_install_spotifyd(ctx):
    """Install and enable the spotifyd daemon."""
    changed = False
    if not ctx.which("spotifyd"):
        ctx.run("curl", "-fsSL", "-o", "/tmp/spotifyd.tar.gz", SPOTIFYD_URL)
        ctx.run("tar", "-xzf", "/tmp/spotifyd.tar.gz", "-C", "/usr/bin")
        os.unlink("/tmp/spotifyd.tar.gz")
        changed = True

    changed |= ctx.write_file(
        "/etc/systemd/system/spotifyd.service",
        SPOTIFYD_SERVICE,
        mode=0o755,
        owner="root",
        group="root",
    )
    if changed:
        ctx.run("systemctl", "daemon-reload")
    changed |= ctx.systemd("spotifyd", enable=True, start=True)
    return changed


register_task("speaker_install_spotifyd", speaker_install_spotifyd, root=True)
