"""Core framework for the provisioning tool.

This module provides:

* :class:`Context` -- the object passed to every task, giving access to the
  current mode (root/user), the invoking user, and helpers for running
  commands and checking whether things are already installed.
* :func:`register_task` -- the function used to register a task.
* :func:`get_task` / :func:`all_tasks` -- registry lookups.

Tasks live in separate modules (see the ``tasks/`` package).  Importing a task
module registers its tasks as a side effect.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


class ProvisionError(RuntimeError):
    """Raised when a task cannot complete successfully."""


#: Statuses a task run can report.
SUCCESS = "success"  # the task made changes
NOT_MODIFIED = "not modified"  # the task had nothing to do
SKIPPED = "skipped"  # the task was skipped (wrong privilege mode)
FAILED = "failed"  # the task raised :class:`ProvisionError`


class Task:
    """A registered provisioning task."""

    def __init__(self, name, func, description="", root=None):
        self.name = name
        self.func = func
        self.description = description
        # Whether the task must run as root (True), as the user (False), or
        # either (None).  Used to validate that a task runs in the right mode.
        self.root = root


# Global task registry, keyed by task name.
_REGISTRY = {}


def register_task(name, func, root=None):
    """Register ``func`` as the task called ``name``.

    ``func`` is called with a single :class:`Context` argument.  Pass
    ``root=True`` if the task must run as root, ``root=False`` if it must run
    as the invoking user, or leave it as ``None`` if it may run either way::

        register_task("install_spotify", install_spotify, root=True)

    A task returns True if it modified the system and False (or None) if it
    had nothing to do.  :func:`run_task` maps the return value to a status.
    """
    if name in _REGISTRY:
        raise ProvisionError(f"task {name!r} is already registered")
    description = ""
    if func.__doc__:
        description = func.__doc__.strip().splitlines()[0]
    _REGISTRY[name] = Task(
        name=name,
        func=func,
        description=description,
        root=root,
    )


def get_task(name):
    """Look up a registered task by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ProvisionError(f"unknown task: {name!r}") from None


def all_tasks():
    """Return all registered tasks, sorted by name."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


class Context:
    """Execution context handed to every task."""

    def __init__(self, root, user, home, verbose=False, options=None):
        # True when running as root, False when running as the invoking user.
        self.root = root
        # The user the script was invoked as (the "real" user, even under sudo).
        self.user = user
        # Home directory of ``user``.
        self.home = home
        # Whether to print the commands that are executed.
        self.verbose = verbose
        # Extra key/value options passed on the command line.
        self.options = options if options is not None else {}

    # -- command helpers ---------------------------------------------------

    def run(self, *args, check=True, capture=False, cwd=None, env=None):
        """Run a command, raising :class:`ProvisionError` on failure.

        When ``capture`` is true the command's stdout is captured and returned
        on the result; otherwise output is streamed to the terminal.
        """
        cmd = [str(a) for a in args]
        if self.verbose:
            print(f"    $ {' '.join(cmd)}", file=sys.stderr)
        try:
            return subprocess.run(
                cmd,
                check=check,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
            )
        except FileNotFoundError as exc:
            raise ProvisionError(f"command not found: {cmd[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() if capture else ""
            raise ProvisionError(
                f"command failed ({exc.returncode}): {' '.join(cmd)}"
                + (f"\n{detail}" if detail else "")
            ) from exc

    def run_ok(self, *args, **kwargs):
        """Run a command and return whether it succeeded (no exception)."""
        try:
            self.run(*args, check=True, capture=True, **kwargs)
            return True
        except ProvisionError:
            return False

    def output(self, *args, **kwargs):
        """Run a command and return its stripped stdout."""
        result = self.run(*args, capture=True, **kwargs)
        return (result.stdout or "").strip()

    # -- package helpers ---------------------------------------------------

    def apt_installed(self, package):
        """Return True if an APT package is installed."""
        result = self.run(
            "dpkg-query", "-W", "-f=${Status}", package, check=False, capture=True
        )
        return result.returncode == 0 and "install ok installed" in (
            result.stdout or ""
        )

    def apt_install(self, *packages, update=False):
        """Install one or more APT packages (idempotent).

        Returns True if anything was installed, False if every package was
        already present.
        """
        missing = [p for p in packages if not self.apt_installed(p)]
        if not missing:
            return False
        if update:
            self.run("apt-get", "update")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        self.run(
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            *missing,
            env=env,
        )
        return True

    def apt_install_deb(self, url_or_path):
        """Install a .deb from a URL or local path.

        Returns True (the install ran).  Callers that want to report
        "not modified" should guard on whether the binary is already present.
        """
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        if "://" not in url_or_path:
            self.run("apt-get", "install", "-y", os.path.abspath(url_or_path), env=env)
            return True
        # apt-get cannot fetch remote .deb files, so download it first.
        with tempfile.TemporaryDirectory(prefix="provision-") as tmpdir:
            # Let the unprivileged _apt user read the package.
            os.chmod(tmpdir, 0o755)
            path = os.path.join(tmpdir, "package.deb")
            self.download(url_or_path, path, mode=0o644)
            self.run("apt-get", "install", "-y", path, env=env)
        return True

    def apt_add_repo(
        self,
        *,
        name,
        key_url=None,
        key_path=None,
        key_dearmor=True,
        repo=None,
        sources_content=None,
    ):
        """Add an APT repository and its signing key (idempotent).

        ``key_url`` is downloaded and (by default) dearmored into
        ``key_path``.  Either ``repo`` (a one-line ``deb ...`` entry) or
        ``sources_content`` (a full deb822 ``.sources`` file) must be given.

        Returns True if the repository state changed (key downloaded or file
        written).
        """
        changed = False
        if key_url and key_path:
            if not os.path.exists(key_path):
                with tempfile.TemporaryDirectory(prefix="provision-") as tmpdir:
                    tmp = os.path.join(tmpdir, f"{name}.key")
                    self.download(key_url, tmp)
                    os.makedirs(os.path.dirname(key_path), exist_ok=True)
                    if key_dearmor:
                        with open(tmp, "rb") as src, open(key_path, "wb") as dst:
                            subprocess.run(
                                ["gpg", "--dearmor"], stdin=src, stdout=dst, check=True
                            )
                    else:
                        shutil.copyfile(tmp, key_path)
                    os.chmod(key_path, 0o644)
                changed = True

        if sources_content is not None:
            path = f"/etc/apt/sources.list.d/{name}.sources"
            changed |= self.write_file(path, sources_content, mode=0o644)
        elif repo is not None:
            path = f"/etc/apt/sources.list.d/{name}.list"
            changed |= self.write_file(path, repo + "\n", mode=0o644)
        return changed

    # -- file helpers ------------------------------------------------------

    def download(self, url, path, *, mode=0o644):
        """Download ``url`` to ``path`` (without relying on curl/wget)."""
        if self.verbose:
            print(f"    $ download {url} -> {path}", file=sys.stderr)
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) provision"}
        )
        try:
            with urllib.request.urlopen(request) as response, open(path, "wb") as fh:
                shutil.copyfileobj(response, fh)
        except OSError as exc:
            raise ProvisionError(f"download failed: {url}: {exc}") from exc
        os.chmod(path, mode)

    def write_file(self, path, content, *, mode=0o644, owner=None, group=None):
        """Write ``content`` to ``path`` if it differs.  Returns True if changed."""
        try:
            with open(path, "r") as fh:
                if fh.read() == content:
                    return False
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        os.chmod(path, mode)
        if owner is not None or group is not None:
            shutil.chown(path, user=owner, group=group)
        return True

    def line_in_file(self, path, *, regexp, line=None, insert_before=None):
        """Ensure a line matching ``regexp`` exists (or is removed).

        If ``line`` is None the matching line is removed.  Otherwise the
        matching line is replaced, or inserted before the first line matching
        ``insert_before`` (or appended).  Returns True if the file changed.
        """
        import re

        try:
            with open(path, "r") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            lines = []

        pattern = re.compile(regexp)
        changed = False
        new_lines = []
        matched = False
        for existing in lines:
            if pattern.search(existing):
                matched = True
                if line is None:
                    changed = True
                    continue
                if existing != line:
                    changed = True
                new_lines.append(line)
            else:
                new_lines.append(existing)

        if line is not None and not matched:
            if insert_before is not None:
                before = re.compile(insert_before)
                for i, existing in enumerate(new_lines):
                    if before.search(existing):
                        new_lines.insert(i, line)
                        break
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            changed = True

        if changed:
            self.write_file(path, "\n".join(new_lines) + "\n")
        return changed

    # -- misc helpers ------------------------------------------------------

    def which(self, program):
        """Return the path to ``program`` if it is on PATH."""
        return shutil.which(program)

    def systemd(self, unit, *, enable=True, start=True):
        """Enable and/or start a systemd unit (idempotent).

        Returns True if the unit state changed.
        """
        changed = False
        if enable:
            enabled = (
                self.run("systemctl", "is-enabled", unit, check=False, capture=True)
                .returncode
                == 0
            )
            if not enabled:
                self.run("systemctl", "enable", unit)
                changed = True
        if start:
            active = self.run(
                "systemctl", "is-active", unit, check=False, capture=True
            ).stdout.strip()
            if active != "active":
                self.run("systemctl", "start", unit)
                changed = True
        return changed

    def dconf(self, key, value):
        """Set a dconf key for the invoking user (idempotent).

        Returns True if the value changed.
        """
        current = self.run(
            "dconf", "read", key, check=False, capture=True
        ).stdout.strip()
        if current == value:
            return False
        self.run("dconf", "write", key, value)
        return True
