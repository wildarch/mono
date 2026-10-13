"""Visual Studio Code installation and configuration tasks."""

import os

from framework import register_task

VSCODE_EXTENSIONS = [
    "vscodevim.vim",
    "james-yu.latex-workshop",
    "llvm-vs-code-extensions.vscode-clangd",
    "ms-vscode-remote.remote-containers",
]

VSCODE_SETTINGS = """\
{
    "extensions.ignoreRecommendations": true,
    "workbench.startupEditor": "none"
}
"""


def install_vscode(ctx):
    """Install VS Code from the official .deb package."""
    if ctx.which("code"):
        return False
    # Pre-seed debconf so the Microsoft repo is added automatically.
    ctx.run(
        "bash",
        "-c",
        "echo 'code code/add-microsoft-repo boolean true' | debconf-set-selections",
    )
    ctx.apt_install_deb(
        "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
    )
    return True


register_task("install_vscode", install_vscode, root=True)


def install_vscode_extensions(ctx):
    """Install the VS Code extensions used by the user."""
    if not ctx.which("code"):
        return False
    installed = ctx.output("code", "--list-extensions").splitlines()
    installed = {line.strip().lower() for line in installed}
    changed = False
    for ext in VSCODE_EXTENSIONS:
        if ext.lower() in installed:
            continue
        ctx.run("code", "--force", "--install-extension", ext)
        changed = True
    return changed


register_task("install_vscode_extensions", install_vscode_extensions, root=False)


def configure_vscode(ctx):
    """Write the VS Code user settings."""
    path = os.path.join(ctx.home, ".config", "Code", "User", "settings.json")
    return ctx.write_file(path, VSCODE_SETTINGS, owner=ctx.user)


register_task("configure_vscode", configure_vscode, root=False)
