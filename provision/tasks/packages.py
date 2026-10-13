"""Package installation tasks."""

from framework import register_task

#: Base packages installed on the laptops (mirrors the "Install APT Packages"
#: task in ``ansible/thinkpad.yml`` and ``ansible/zbook.yml``).
LAPTOP_PACKAGES = [
    "spotify-client",
    "build-essential",
    "default-jdk",
    "git",
    "vim-gtk3",
    "gimp",
    "inkscape",
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
]


def install_apt_packages(ctx):
    """Install the base set of APT packages for the laptops."""
    return ctx.apt_install(*LAPTOP_PACKAGES, update=True)


register_task("install_apt_packages", install_apt_packages, root=True)


def install_docker(ctx):
    """Install Docker and add the invoking user to the docker group."""
    changed = False
    if not ctx.which("docker"):
        ctx.run("apt-get", "update")
        changed = ctx.apt_install("docker.io")
    changed |= ctx.systemd("docker", enable=True, start=True)
    # Add the user to the docker group (idempotent).
    groups = ctx.output("groups", ctx.user)
    if "docker" not in groups.split():
        ctx.run("usermod", "-aG", "docker", ctx.user)
        changed = True
    return changed


register_task("install_docker", install_docker, root=True)


def set_default_editor(ctx):
    """Use vim.gtk3 as the default editor."""
    if not ctx.which("vim.gtk3"):
        return False
    current = ctx.output(
        "readlink", "-f", "/etc/alternatives/editor", check=False
    )
    if current == "/usr/bin/vim.gtk3":
        return False
    ctx.run(
        "update-alternatives",
        "--set",
        "editor",
        "/usr/bin/vim.gtk3",
    )
    return True


register_task("set_default_editor", set_default_editor, root=True)
