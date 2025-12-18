# mono
My own personal monorepo.

# Bootstrapping
```
# 1. Install bitwarden desktop (for SSH keys)
snap install bitwarden

# 2. Log in to bitwarden
# In the app settings, set 'Enable SSH agent' to YES
echo 'export SSH_AUTH_SOCK=$HOME/snap/bitwarden/current/.bitwarden-ssh-agent.sock' >> ~/.bashrc
source ~/.bashrc

# 3. Download git
sudo apt install git

# 4. Clone the repository
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:wildarch/mono.git

# 5. Run commands below
sudo apt install ansible
ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/thinkpad.yml -K

# 6. Configure git identity
git config --global user.email "daandegraaf9@gmail.com"
git config --global user.name "Daan de Graaf"
```
