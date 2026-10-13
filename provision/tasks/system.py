"""Miscellaneous system configuration tasks."""

from framework import register_task


def install_zoom(ctx):
    """Install Zoom from the official .deb package."""
    if ctx.which("zoom"):
        return False
    ctx.apt_install_deb("https://zoom.us/client/latest/zoom_amd64.deb")
    return True


register_task("install_zoom", install_zoom, root=True)


def configure_lid_switch(ctx):
    """Configure lid-switch behaviour (ZBook).

    Shut down when the lid is closed on battery, ignore it on AC power.
    """
    changed = False
    changed |= ctx.line_in_file(
        "/etc/systemd/logind.conf",
        regexp=r"^HandleLidSwitch=",
        line="HandleLidSwitch=poweroff",
    )
    changed |= ctx.line_in_file(
        "/etc/systemd/logind.conf",
        regexp=r"^HandleLidSwitchExternalPower=",
        line="HandleLidSwitchExternalPower=ignore",
    )
    if changed:
        ctx.run("systemctl", "restart", "systemd-logind")
    return changed


register_task("configure_lid_switch", configure_lid_switch, root=True)
