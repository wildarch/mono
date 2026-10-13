"""GNOME desktop configuration tasks (mirrors ``ansible/tasks/gnome.yml``)."""

from framework import register_task

#: dconf key/value pairs applied for the user.
GNOME_SETTINGS = [
    ("/org/gnome/settings-daemon/plugins/media-keys/volume-mute", "['F1']"),
    ("/org/gnome/settings-daemon/plugins/media-keys/volume-down", "['F2']"),
    ("/org/gnome/settings-daemon/plugins/media-keys/volume-up", "['F3']"),
    ("/org/gnome/settings-daemon/plugins/power/ambient-enabled", "false"),
    # Read the current set with: dconf read /org/gnome/shell/favorite-apps
    (
        "/org/gnome/shell/favorite-apps",
        "['firefox_firefox.desktop', 'org.gnome.Ptyxis.desktop', 'com.microsoft.VSCode.desktop', 'spotify.desktop', 'org.gnome.Nautilus.desktop']"
    ),
    (
        "/org/gnome/desktop/wm/keybindings/switch-windows",
        "['<Alt>Tab']",
    ),
    (
        "/org/gnome/desktop/wm/keybindings/switch-applications",
        "@as []",
    ),
    (
        "/org/gnome/desktop/input-sources/sources",
        "[('xkb', 'us'), ('xkb', 'us+intl')]",
    ),
    (
        "/org/gnome/settings-daemon/plugins/color/night-light-enabled",
        "true",
    ),
]


def configure_gnome(ctx):
    """Apply GNOME desktop settings via dconf."""
    if not ctx.which("dconf"):
        return False
    changed = False
    for key, value in GNOME_SETTINGS:
        changed |= ctx.dconf(key, value)
    return changed


register_task("configure_gnome", configure_gnome, root=False)
