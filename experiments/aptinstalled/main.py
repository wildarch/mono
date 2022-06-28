#!/usr/bin/env python3
import apt
import yaml

def get_installed_packages():
    cache = apt.Cache()

    for p in cache:
        pkg = cache[p]
        if pkg.is_installed and not pkg.is_auto_installed and not pkg.essential:
            yield pkg.name

def get_playbook_packages():
    for playbook_path in ["ansible/yoga.yml", "ansible/build.yml"]:
        with open(playbook_path) as f:
            playbooks = yaml.load(f, Loader=yaml.FullLoader)
            for playbook in playbooks:
                for task in playbook["tasks"]:
                    if "apt" in task:
                        apt = task["apt"]
                        if "pkg" in apt:
                            for pkg in apt["pkg"]:
                                yield pkg

def print_packages(msg, pkgs):
    if len(pkgs) == 0:
        print(f"{msg}: none")
    else:
        print(msg)
        for pkg in sorted(pkgs):
            print(pkg)
        print()

def get_base_pkgs():
    with open("experiments/aptinstalled/basepkgs.txt") as f:
        for line in f:
            yield line.rstrip()

base = set(get_base_pkgs())
installed = set(get_installed_packages())
playbook = set(get_playbook_packages())

print_packages("In playbook but not installed", playbook - installed)
print_packages("Installed but not in playbook", installed - playbook)

print_packages("Installed but not in playbook or base packages", installed - playbook - base)