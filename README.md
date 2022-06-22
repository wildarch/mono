# mono
My own personal monorepo.

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
```
