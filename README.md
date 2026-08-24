# mono
My own personal monorepo.

# Bootstrapping
```bash
# Install KeepassXC (for SSH keys)
sudo apt install keepassxc

# Copy the KeepassXC database to your home directory.
# Unlock it, then enable the SSH Agent (https://keepassxc.org/docs/KeePassXC_UserGuide#_setup_ssh_agent_integration)

# Download git
sudo apt install git

# Clone the repository
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:wildarch/mono.git

# Run commands below
sudo apt install ansible
ansible-galaxy install -r ansible/requirements.yml
export ANSIBLE_BECOME_EXE=sudo.ws # Only on Ubuntu 26.04 (sudo-rs and ansible are incompatible)
ansible-playbook ansible/thinkpad.yml -K

# Configure git identity
git config --global user.email "daandegraaf9@gmail.com"
git config --global user.name "Daan de Graaf"
```