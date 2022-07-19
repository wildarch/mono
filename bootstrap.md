# Bootstrapping
```
# 1. Add user to sudo group
su
/usr/sbin/usermod -aG sudo daan

# 2. Reboot for changes to take effect
/usr/sbin/reboot

# 3. Download git
sudo apt install git

# 4. Clone the repository
git clone https://github.com/wildarch/mono.git 

# 5. Run commands below
sudo apt install ansible
ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/yoga.yml -K

# 6. Install dependencies for building the monorepo
ansible-playbook ansible/build.yml -K

$ 7. Restore personal configuration
bazel run //configsync restore

# 8. Once SSH Keys are setup, change to an SSH url:
git remote set-url origin git@github.com:wildarch/mono.git
```