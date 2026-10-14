"""APT repository setup tasks.

These mirror the ``ansible/tasks/*.yml`` files that add third-party APT
repositories.  Each task is idempotent: it only downloads keys and writes
source files when they are missing or out of date.
"""

from framework import register_task


def _codename(ctx):
    """Return the Ubuntu codename (e.g. ``noble``)."""
    with open("/etc/os-release") as fh:
        for line in fh:
            if line.startswith("VERSION_CODENAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return ctx.output("lsb_release", "-cs")

def install_eduvpn_repo(ctx):
    """Add the EduVPN APT repository."""
    return ctx.apt_add_repo(
        name="eduvpn-v4",
        key_url="https://app.eduvpn.org/linux/v4/deb/app+linux@eduvpn.org.asc",
        key_path="/usr/share/keyrings/eduvpn-v4.gpg",
        repo=(
            "deb [arch=amd64 signed-by=/usr/share/keyrings/eduvpn-v4.gpg] "
            "https://app.eduvpn.org/linux/v4/deb/ resolute main"
        ),
    )


register_task("install_eduvpn_repo", install_eduvpn_repo, root=True)


def install_zotero_repo(ctx):
    """Add the Zotero APT repository."""
    key_path = "/usr/share/keyrings/zotero-archive-keyring.gpg"
    return ctx.apt_add_repo(
        name="zotero",
        key_url=(
            "https://raw.githubusercontent.com/retorquere/zotero-deb/"
            "master/zotero-archive-keyring.asc"
        ),
        key_path=key_path,
        sources_content=(
            "Types: deb\n"
            "URIs: https://zotero.retorque.re/file/apt-package-archive\n"
            "Suites: ./\n"
            "By-Hash: force\n"
            f"Signed-By: {key_path}\n"
        ),
    )


register_task("install_zotero_repo", install_zotero_repo, root=True)


def install_displaylink_repo(ctx):
    """Install the Synaptics (DisplayLink) repository keyring."""
    if ctx.apt_installed("synaptics-repository-keyring"):
        return False
    ctx.apt_install_deb(
        "https://www.synaptics.com/sites/default/files/Ubuntu/pool/stable/"
        "main/all/synaptics-repository-keyring.deb"
    )
    return True


register_task("install_displaylink_repo", install_displaylink_repo, root=True)


def install_spotify_repo(ctx):
    """Add the Spotify APT repository."""
    return ctx.apt_add_repo(
        name="spotify",
        key_url="https://download.spotify.com/debian/pubkey_5384CE82BA52C83A.asc",
        key_path="/etc/apt/trusted.gpg.d/spotify.gpg",
        repo="deb https://repository.spotify.com stable non-free",
    )


register_task("install_spotify_repo", install_spotify_repo, root=True)


def install_tailscale_repo(ctx):
    """Add the Tailscale APT repository."""
    codename = _codename(ctx)
    return ctx.apt_add_repo(
        name="tailscale",
        key_url=(
            f"https://pkgs.tailscale.com/stable/ubuntu/{codename}.noarmor.gpg"
        ),
        key_path="/usr/share/keyrings/tailscale-archive-keyring.gpg",
        key_dearmor=False,
        repo=(
            "deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.gpg] "
            f"https://pkgs.tailscale.com/stable/ubuntu {codename} main"
        ),
    )


register_task("install_tailscale_repo", install_tailscale_repo, root=True)
