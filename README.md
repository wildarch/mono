# mono
My own personal monorepo.

# Bootstrapping
```bash
# Install KeepassXC (for SSH keys)
sudo apt install keepassxc

# Copy the KeepassXC database to your home directory.
# Unlock it, then enable the SSH Agent
# See https://keepassxc.org/docs/KeePassXC_UserGuide#_setup_ssh_agent_integration
# Then, right-click on the SSH keys and click 'Add key to SSH agent'

# Download git
sudo apt install git

# Clone the repository
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:wildarch/mono.git

sudo ./provision/provision.py --system thinkpad
./provision/provision.py --system thinkpad

# Configure git identity
git config --global user.email "daandegraaf9@gmail.com"
git config --global user.name "Daan de Graaf"
```