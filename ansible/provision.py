#!/usr/bin/env python3
"""Provision a fresh Thinkpad install without Ansible.

Reproduces the steps from ``ansible/thinkpad.yml`` (and the task files it
includes) as a standalone, idempotent Python script.

Usage:
    sudo python3 provision.py                 # run every step
    python3 provision.py                      # same, but prompts for sudo once
    python3 provision.py --step apt_packages  # run a single step
    python3 provision.py --list-steps         # print available steps

Steps that need root are re-launched as a copy of this script via ``sudo``.
The top-level run prompts for the sudo password once at startup and reuses it
for every root step.
"""

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Packages installed via apt (from thinkpad.yml).
APT_PACKAGES = [
    "spotify-client",
    "build-essential",
    "default-jdk",
    "git",
    "vim-gtk3",
    "gimp",
    "inkscape",
    "python3-psutil",  # Allows setting keyboard shortcuts
    "sshpass",  # To use ask-ssh-pass with Ansible
    "bash-completion",  # Enable bash completion
    "zotero",
    # Latex related
    "latexmk",
    "texlive-latex-recommended",
    "texlive-latex-extra",
    "texlive-fonts-recommended",
    "texlive-fonts-extra",
    "texlive-science",
    "texlive-extra-utils",
    # End Latex related
    "ubuntu-restricted-extras",  # For MP3 support
    "totem",  # Media player
    "eduvpn-client",
    "displaylink-driver",
    "tailscale",
    "keepassxc",
    "docker.io",  # Replaces the geerlingguy.docker role (no compose)
]

# VS Code extensions to install.
VSCODE_EXTENSIONS = [
    "vscodevim.vim",
    "james-yu.latex-workshop",
    "llvm-vs-code-extensions.vscode-clangd",
    "ms-vscode-remote.remote-containers",
]

VSCODE_SETTINGS = {
    "extensions.ignoreRecommendations": True,
    "workbench.startupEditor": "none",
}

# GNOME dconf settings (schema, key, value) from tasks/gnome.yml.
GNOME_SETTINGS = [
    ("org.gnome.settings-daemon.plugins.media-keys", "volume-mute", ["F1"]),
    ("org.gnome.settings-daemon.plugins.media-keys", "volume-down", ["F2"]),
    ("org.gnome.settings-daemon.plugins.media-keys", "volume-up", ["F3"]),
    ("org.gnome.settings-daemon.plugins.power", "ambient-enabled", False),
    (
        "org.gnome.shell",
        "favorite-apps",
        [
            "firefox_firefox.desktop",
            "spotify.desktop",
            "org.gnome.Terminal.desktop",
            "org.gnome.Nautilus.desktop",
            "code.desktop",
        ],
    ),
    ("org.gnome.desktop.wm.keybindings", "switch-windows", ["<Alt>Tab"]),
    ("org.gnome.desktop.wm.keybindings", "switch-applications", []),
    ("org.gnome.desktop.input-sources", "sources", [("xkb", "us"), ("xkb", "us+intl")]),
    ("org.gnome.settings-daemon.plugins.color", "night-light-enabled", True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"==> {msg}", flush=True)


def run(cmd, check=True, input=None, **kwargs):
    """Run a command, streaming output. Returns CompletedProcess."""
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, input=input, **kwargs)


def download(url, dest):
    """Download ``url`` to ``dest`` (overwrites)."""
    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def apt_install(packages):
    """Install apt packages, skipping ones already installed."""
    import apt

    cache = apt.Cache()
    cache.update()
    cache.open(None)

    to_install = []
    for name in packages:
        pkg = cache[name]
        if pkg.is_installed:
            log(f"Already installed: {name}")
        else:
            to_install.append(pkg)

    if not to_install:
        log("All apt packages already installed.")
        return

    log(f"Installing: {', '.join(p.name for p in to_install)}")
    for pkg in to_install:
        pkg.mark_install()
    cache.commit()


def install_deb(url):
    """Download and install a .deb package."""
    with tempfile.NamedTemporaryFile(suffix=".deb", delete=False) as tmp:
        deb_path = tmp.name
    try:
        download(url, deb_path)
        run(["apt-get", "install", "-y", deb_path])
    finally:
        os.unlink(deb_path)


def add_apt_source(filename, content):
    """Write an apt source file, skipping if content is unchanged."""
    path = f"/etc/apt/sources.list.d/{filename}"
    if os.path.exists(path) and open(path).read() == content:
        log(f"Source already present: {path}")
        return
    log(f"Writing {path}")
    with open(path, "w") as fh:
        fh.write(content)


def install_gpg_key(url, dest):
    """Download a GPG key and dearmor it into ``dest`` (skip if present)."""
    if os.path.exists(dest):
        log(f"Key already present: {dest}")
        return
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
        asc_path = tmp.name
    try:
        download(url, asc_path)
        # gpg --dearmor reads the .asc from stdin and writes the keyring to stdout.
        result = subprocess.run(
            ["gpg", "--dearmor"],
            input=open(asc_path, "rb").read(),
            capture_output=True,
            check=True,
        )
        with open(dest, "wb") as fh:
            fh.write(result.stdout)
        log(f"Wrote keyring {dest}")
    finally:
        os.unlink(asc_path)


def set_dconf(schema, key, value):
    """Set a dconf key via Gio.Settings, skipping if already equal."""
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio

    settings = Gio.Settings.new(schema)
    current = settings.get_value(key)
    if current == value:
        log(f"dconf already set: {schema} {key}")
        return
    log(f"Setting dconf: {schema} {key} = {value!r}")
    settings.set_value(key, value)
    settings.apply()


def bootstrap():
    """Ensure python3-apt and python3-gi are importable.

    These are imported lazily by the step functions, so a fresh system may
    lack them. Rather than installing automatically, report a clear error.
    """
    missing = []
    for module, pkg in (("apt", "python3-apt"), ("gi", "python3-gi")):
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        sys.exit(
            "Missing required Python modules: "
            + ", ".join(missing)
            + "\nInstall them with: sudo apt install python3-apt python3-gi"
        )
    log("Bootstrap: python3-apt and python3-gi are present.")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_eduvpn_repo():
    """Add the EduVPN apt repository."""
    key_url = "https://app.eduvpn.org/linux/v4/deb/app+linux@eduvpn.org.asc"
    key_dest = "/usr/share/keyrings/eduvpn-v4.gpg"
    install_gpg_key(key_url, key_dest)
    add_apt_source(
        "eduvpn-v4.list",
        "deb [arch=amd64 signed-by=/usr/share/keyrings/eduvpn-v4.gpg] "
        "https://app.eduvpn.org/linux/v4/deb/ resolute main\n",
    )


def step_zotero_repo():
    """Add the Zotero apt repository."""
    key_url = (
        "https://raw.githubusercontent.com/retorquere/zotero-deb/master/"
        "zotero-archive-keyring.asc"
    )
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
        asc_path = tmp.name
    try:
        download(key_url, asc_path)
        # Format the key exactly like the original install script.
        key = open(asc_path).read()
        key = key.replace("\n\n", "\n.\n")  # s/^$/./ on blank lines
        key = "\n".join(" " + line if line else "." for line in key.split("\n"))
        content = (
            "Types: deb\n"
            "URIs: https://zotero.retorque.re/file/apt-package-archive\n"
            "Suites: ./\n"
            "By-Hash: force\n"
            f"Signed-By:{key}\n"
        )
        add_apt_source("zotero.sources", content)
    finally:
        os.unlink(asc_path)


def step_spotify_repo():
    """Add the Spotify apt repository."""
    key_url = "https://download.spotify.com/debian/pubkey_5384CE82BA52C83A.asc"
    key_dest = "/etc/apt/trusted.gpg.d/spotify.gpg"
    install_gpg_key(key_url, key_dest)
    add_apt_source(
        "spotify.list", "deb https://repository.spotify.com stable non-free\n"
    )


def step_tailscale_repo():
    """Add the Tailscale apt repository."""
    release = subprocess.run(
        ["lsb_release", "-cs"], capture_output=True, text=True, check=True
    ).stdout.strip()
    key_url = f"https://pkgs.tailscale.com/stable/ubuntu/{release}.noarmor.gpg"
    key_dest = "/usr/share/keyrings/tailscale-archive-keyring.gpg"
    install_gpg_key(key_url, key_dest)
    add_apt_source(
        "tailscale.list",
        f"deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] "
        f"https://pkgs.tailscale.com/stable/ubuntu {release} main\n",
    )


def step_displaylink_repo():
    """Install the Synaptics (DisplayLink) repository keyring."""
    install_deb(
        "https://www.synaptics.com/sites/default/files/Ubuntu/pool/stable/"
        "main/all/synaptics-repository-keyring.deb"
    )


def step_vscode_repo():
    """Set debconf and install VS Code."""
    run(
        [
            "debconf-set-selections",
            "code code/add-microsoft-repo boolean true",
        ]
    )
    install_deb(
        "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
    )


def step_zoom_repo():
    """Install Zoom."""
    install_deb("https://zoom.us/client/latest/zoom_amd64.deb")


def step_apt_packages():
    """Install all apt packages from the playbook."""
    apt_install(APT_PACKAGES)


def step_editor_alternative():
    """Set vim.gtk3 as the default editor."""
    run(["update-alternatives", "--set", "editor", "/usr/bin/vim.gtk3"])


def step_vscode_extensions():
    """Install VS Code extensions."""
    installed = subprocess.run(
        ["code", "--list-extensions"], capture_output=True, text=True
    ).stdout.split()
    for ext in VSCODE_EXTENSIONS:
        if ext in installed:
            log(f"Extension already installed: {ext}")
        else:
            run(["code", "--force", "--install-extension", ext])


def step_vscode_settings():
    """Write VS Code user settings."""
    settings_dir = Path.home() / ".config" / "Code" / "User"
    settings_path = settings_dir / "settings.json"
    content = json.dumps(VSCODE_SETTINGS, indent=4) + "\n"
    if settings_path.exists() and settings_path.read_text() == content:
        log(f"VS Code settings already present: {settings_path}")
        return
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(content)
    log(f"Wrote {settings_path}")


def step_docker_group():
    """Add the user to the docker group."""
    user = "daan"
    groups = subprocess.run(
        ["getent", "group", "docker"], capture_output=True, text=True
    ).stdout
    if user in groups.split(":")[-1].split(","):
        log(f"{user} already in docker group.")
        return
    run(["usermod", "-aG", "docker", user])


def step_gnome_settings():
    """Apply GNOME dconf settings."""
    for schema, key, value in GNOME_SETTINGS:
        set_dconf(schema, key, value)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    ("eduvpn_repo", step_eduvpn_repo, True),
    ("zotero_repo", step_zotero_repo, True),
    ("spotify_repo", step_spotify_repo, True),
    ("tailscale_repo", step_tailscale_repo, True),
    ("displaylink_repo", step_displaylink_repo, True),
    ("vscode_repo", step_vscode_repo, True),
    ("zoom_repo", step_zoom_repo, True),
    ("apt_packages", step_apt_packages, True),
    ("editor_alternative", step_editor_alternative, True),
    ("vscode_extensions", step_vscode_extensions, False),
    ("vscode_settings", step_vscode_settings, False),
    ("docker_group", step_docker_group, True),
    ("gnome_settings", step_gnome_settings, False),
]

STEP_NAMES = [name for name, _, _ in STEPS]


def run_step(name):
    """Run a single step by name."""
    for step_name, func, requires_root in STEPS:
        if step_name == name:
            if requires_root and os.geteuid() != 0:
                sys.exit(
                    f"Step '{name}' requires root; run via sudo or the top-level script."
                )
            log(f"Running step: {name}")
            func()
            return
    sys.exit(f"Unknown step: {name}. Use --list-steps to see available steps.")


def run_all(sudo_password):
    """Run every step, re-launching root steps via sudo."""
    for name, _, requires_root in STEPS:
        if requires_root:
            if os.geteuid() == 0:
                run_step(name)
            else:
                log(f"Running step (as root): {name}")
                run(
                    [
                        "sudo",
                        "-S",
                        "-p",
                        "",
                        sys.executable,
                        os.path.abspath(__file__),
                        "--step",
                        name,
                    ],
                    input=(sudo_password + "\n").encode(),
                )
        else:
            run_step(name)


def main():
    parser = argparse.ArgumentParser(description="Provision a fresh Thinkpad install.")
    parser.add_argument(
        "--step",
        metavar="NAME",
        help="Run a single provisioning step (see --list-steps).",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available steps and exit.",
    )
    args = parser.parse_args()

    if args.list_steps:
        for name, _, requires_root in STEPS:
            marker = " (root)" if requires_root else ""
            print(f"{name}{marker}")
        return

    # The script must never be run as root for a full run: root steps are
    # re-launched individually via `sudo ... --step <name>`. Running the whole
    # thing as root would bypass the sudo-per-step design.
    if os.geteuid() == 0 and not args.step:
        sys.exit(
            "Refusing to run as root without --step.\n"
            "Run the script as a normal user (it will prompt for sudo), or "
            "use `sudo python3 provision.py --step <name>` for a single step."
        )

    # Full run: prompt for sudo once, then run all steps.
    if os.geteuid() != 0:
        sudo_password = getpass.getpass("sudo password: ")
    else:
        sudo_password = None

    bootstrap()

    if args.step:
        run_step(args.step)
        return

    run_all(sudo_password)


if __name__ == "__main__":
    main()
