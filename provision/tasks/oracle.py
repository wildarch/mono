"""Oracle VM tasks (mirrors ``ansible/oracle/oracle.yml``).

The Caddy configuration requires a hashed basic-auth password.  Pass it with
``-e caddy_auth_password_hashed=<hash>`` (generate it with
``ssh oracle caddy hash-password``).
"""

import os

from framework import ProvisionError, register_task

HEADSCALE_VERSION = "0.29.4"
HEADSCALE_URL = (
    "https://github.com/juanfont/headscale/releases/download/"
    f"v{HEADSCALE_VERSION}/headscale_{HEADSCALE_VERSION}_linux_arm64.deb"
)

CADDYFILE = """\
{{
\tadmin localhost:2019
}}

oracle.wildarch.dev {{
\troute /cast* {{
\t\tbasicauth {{
\t\t\tdaan {password}
\t\t}}
\t\turi strip_prefix /cast
\t\treverse_proxy 127.0.0.1:8001
\t}}
}}

headscale.wildarch.dev {{
\treverse_proxy localhost:8080
}}
"""


def oracle_install_headscale(ctx):
    """Install and enable headscale."""
    changed = False
    if not ctx.which("headscale"):
        ctx.apt_install_deb(HEADSCALE_URL)
        changed = True
    changed |= ctx.systemd("headscale", enable=True, start=True)
    return changed


register_task("oracle_install_headscale", oracle_install_headscale, root=True)


def oracle_install_caddy(ctx):
    """Install Caddy from the official Cloudsmith repository."""
    if ctx.which("caddy"):
        return ctx.systemd("caddy", enable=True, start=True)
    ctx.apt_install(
        "debian-keyring",
        "debian-archive-keyring",
        "apt-transport-https",
        "curl",
    )
    ctx.run(
        "bash",
        "-c",
        "curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' "
        "| gpg --dearmor > "
        "/usr/share/keyrings/caddy-stable-archive-keyring.gpg",
    )
    ctx.run(
        "bash",
        "-c",
        "curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' "
        "> /etc/apt/sources.list.d/caddy-stable.list",
    )
    os.chmod("/usr/share/keyrings/caddy-stable-archive-keyring.gpg", 0o644)
    os.chmod("/etc/apt/sources.list.d/caddy-stable.list", 0o644)
    ctx.apt_install("caddy", update=True)
    ctx.systemd("caddy", enable=True, start=True)
    return True


register_task("oracle_install_caddy", oracle_install_caddy, root=True)


def oracle_configure_firewall(ctx):
    """Open ports 80 and 443 in the iptables rules."""
    rules = "/etc/iptables/rules.v4"
    if not os.path.exists(rules):
        return
    changed = False
    changed |= ctx.line_in_file(
        rules,
        regexp=r"^-A INPUT -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT",
        line="-A INPUT -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT",
        insert_before=r"^-A INPUT -j REJECT",
    )
    changed |= ctx.line_in_file(
        rules,
        regexp=r"^-A INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT",
        line="-A INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT",
        insert_before=r"^-A INPUT -j REJECT",
    )
    if changed:
        ctx.run("bash", "-c", f"iptables-restore < {rules}")
    return changed


register_task("oracle_configure_firewall", oracle_configure_firewall, root=True)


def oracle_configure_caddy(ctx):
    """Write the Caddyfile and reload Caddy."""
    password = ctx.options.get("caddy_auth_password_hashed")
    if not password:
        raise ProvisionError(
            "missing -e caddy_auth_password_hashed=<hash> "
            "(generate with `caddy hash-password`)"
        )
    changed = ctx.write_file(
        "/etc/caddy/Caddyfile",
        CADDYFILE.format(password=password),
        mode=0o644,
        owner="root",
        group="root",
    )
    if changed:
        ctx.run("systemctl", "reload", "caddy")
    return changed


register_task("oracle_configure_caddy", oracle_configure_caddy, root=True)


def oracle_install_cast_backend(ctx):
    """Install the cast backend .deb (built by ``cast/build_deb.sh``)."""
    if ctx.which("cast-backend"):
        return False
    deb = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cast",
        "cast-backend_0.0.1-1_arm64.deb",
    )
    if not os.path.exists(deb):
        raise ProvisionError(
            f"cast backend package not found at {deb}; "
            "build it with cast/build_deb.sh first"
        )
    ctx.apt_install_deb(deb)
    return True


register_task("oracle_install_cast_backend", oracle_install_cast_backend, root=True)
