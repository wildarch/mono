#!/usr/bin/env python3
"""Provision a fresh install.

This tool installs additional packages and applies system configuration,
mirroring the Ansible configuration under ``ansible/``.

It runs one or more *tasks*.  Each task is a Python function registered with
:func:`framework.register_task` (see the ``tasks/`` package).  Tasks are
idempotent: they detect whether any changes are necessary and return early if
not.

Examples
--------

Run everything needed for the Thinkpad laptop::

    sudo ./provision.py --system thinkpad

Run a single task in isolation::

    sudo ./provision.py --task install_spotify

Some tasks must run as root, others as the current user.  Use ``--root`` and
``--user`` to select the mode explicitly; the UID is asserted to match.  If
neither flag is given, the UID the script runs under selects the mode.
"""

import argparse
import os
import pwd
import sys

# Make sibling modules importable regardless of the current directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework import (  # noqa: E402
    FAILED,
    NOT_MODIFIED,
    SKIPPED,
    SUCCESS,
    Context,
    ProvisionError,
    all_tasks,
    get_task,
)
import tasks  # noqa: E402  (importing registers all tasks)


# ---------------------------------------------------------------------------
# Task collections
# ---------------------------------------------------------------------------
#
# A collection is an ordered list of task names.  ``--system <name>`` runs the
# collection with that name.  Tasks that require a specific privilege level are
# skipped (with a warning) when the current mode does not match, so a single
# collection can be run once as root and once as the user.

COLLECTIONS = {
    "thinkpad": [
        # root tasks
        "install_apt_packages",
        "install_eduvpn_repo",
        "install_zotero_repo",
        "install_displaylink_repo",
        "install_vscode",
        "install_zoom",
        "install_spotify_repo",
        "install_tailscale_repo",
        "install_docker",
        "set_default_editor",
        # user tasks
        "install_vscode_extensions",
        "configure_vscode",
        "configure_gnome",
    ],
    "zbook": [
        "install_apt_packages",
        "install_eduvpn_repo",
        "install_zotero_repo",
        "install_displaylink_repo",
        "install_vscode",
        "install_zoom",
        "install_spotify_repo",
        "install_tailscale_repo",
        "install_docker",
        "set_default_editor",
        "configure_lid_switch",
        "configure_gnome",
        "install_vscode_extensions",
        "configure_vscode",
    ],
    "speaker": [
        "speaker_authorized_keys",
        "speaker_audio_config",
        "speaker_install_spotifyd",
    ],
    "oracle": [
        "oracle_install_headscale",
        "oracle_install_caddy",
        "oracle_configure_firewall",
        "oracle_configure_caddy",
        "oracle_install_cast_backend",
    ],
}


# ---------------------------------------------------------------------------
# Mode handling
# ---------------------------------------------------------------------------


def resolve_mode(args):
    """Return True for root mode, False for user mode.

    Asserts that the requested mode matches the actual UID.
    """
    uid = os.getuid()
    if args.root and args.user:
        raise ProvisionError("--root and --user are mutually exclusive")

    if args.root:
        if uid != 0:
            raise ProvisionError(
                f"--root requested but running as UID {uid} (must be 0)"
            )
        return True
    if args.user:
        if uid == 0:
            raise ProvisionError(
                "--user requested but running as UID 0 (must be > 0)"
            )
        return False

    # Neither flag: select the mode from the UID.
    return uid == 0


def invoking_user():
    """Return (username, home) of the user that invoked the script.

    When run under sudo this is the original user (``SUDO_USER``); otherwise it
    is the current user.
    """
    name = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    try:
        entry = pwd.getpwnam(name)
        return name, entry.pw_dir
    except KeyError:
        return name, os.path.expanduser("~")


# ---------------------------------------------------------------------------
# Running tasks
# ---------------------------------------------------------------------------


def run_task(task_name, ctx, *, strict):
    """Run a single task and return its status.

    The status is one of ``SUCCESS`` (the task made changes),
    ``NOT_MODIFIED`` (nothing to do), ``SKIPPED`` (wrong privilege mode) or
    ``FAILED`` (the task raised :class:`ProvisionError`).
    """
    task = get_task(task_name)

    if task.root is not None and task.root != ctx.root:
        required = "root" if task.root else "user"
        if strict:
            raise ProvisionError(
                f"task {task_name!r} must run as {required} "
                f"(current mode: {'root' if ctx.root else 'user'})"
            )
        print(f"  - {task_name}: skipped (requires {required})")
        return SKIPPED

    try:
        changed = task.func(ctx)
    except ProvisionError as exc:
        print(f"  - {task_name}: failed ({exc})")
        return FAILED

    status = SUCCESS if changed else NOT_MODIFIED
    print(f"  - {task_name}: {status}")
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Provision a fresh install.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--system",
        metavar="NAME",
        help="run the collection of tasks for a system "
        f"({', '.join(sorted(COLLECTIONS))})",
    )
    group.add_argument(
        "--task",
        action="append",
        metavar="NAME",
        help="run a single task in isolation (may be repeated)",
    )
    parser.add_argument(
        "--root",
        action="store_true",
        help="run in root mode (asserts UID == 0)",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        help="run in user mode (asserts UID > 0)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list all registered tasks and exit",
    )
    parser.add_argument(
        "--list-systems",
        action="store_true",
        help="list all task collections and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print each command before running it",
    )
    parser.add_argument(
        "-e",
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra option passed to tasks (may be repeated)",
    )
    args = parser.parse_args(argv)

    if args.list:
        for task in all_tasks():
            mode = {True: "root", False: "user", None: "any"}[task.root]
            print(f"{task.name:32} [{mode:4}] {task.description}")
        return 0

    if args.list_systems:
        for name, names in sorted(COLLECTIONS.items()):
            print(f"{name}:")
            for task_name in names:
                print(f"  - {task_name}")
        return 0

    if not args.system and not args.task:
        parser.error("one of --system or --task is required")

    try:
        root = resolve_mode(args)
    except ProvisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    user, home = invoking_user()
    options = {}
    for item in args.extra:
        if "=" not in item:
            parser.error(f"invalid --extra value (expected KEY=VALUE): {item!r}")
        key, value = item.split("=", 1)
        options[key] = value

    ctx = Context(
        root=root,
        user=user,
        home=home,
        verbose=args.verbose,
        options=options,
    )

    if args.task:
        task_names = args.task
        strict = True
    else:
        if args.system not in COLLECTIONS:
            print(
                f"error: unknown system {args.system!r} "
                f"(known: {', '.join(sorted(COLLECTIONS))})",
                file=sys.stderr,
            )
            return 2
        task_names = COLLECTIONS[args.system]
        strict = False

    print(f"Running as {'root' if root else 'user'} ({user})")
    failures = []
    for task_name in task_names:
        try:
            status = run_task(task_name, ctx, strict=strict)
        except ProvisionError as exc:
            print(f"    error: {exc}", file=sys.stderr)
            failures.append(task_name)
            continue
        if status == FAILED:
            failures.append(task_name)

    if failures:
        print(f"\n{len(failures)} task(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
