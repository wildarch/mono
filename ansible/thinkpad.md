# Setting up Thinkpad
## Backup files
- [ ] Check for uncommitted files in workspace
- [ ] Check for files under ~/Downloads to keep

## Ansible config
Key software to install:
- Google Chrome
- VS Code (+extension config)
- Zotero
- Zoom
- Spotify
- *TODO*: Claude Code
- DisplayLink?

Settings:
- Gnome tweaks
- Other minor deps
- VIM default editor

## Ubuntu
Latest LTS is 24.04.

## System Suspend
TODO: Try it out.

## Synaptics DisplayLink
https://www.synaptics.com/products/displaylink-graphics/downloads/ubuntu

## VS Code
- ms-vscode-remote.remote-containers

## Claude Code
```bash
sudo apt install npm --no-install-recommends

# Setup NPM prefix
mkdir -p ~/.local
npm config set prefix ~/.local
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install Claude Code
npm install -g @anthropic-ai/claude-code

```

## EduVPN
```bash
sudo apt update
sudo apt install apt-transport-https wget
wget -O- https://app.eduvpn.org/linux/v4/deb/app+linux@eduvpn.org.asc | gpg --dearmor | sudo tee /usr/share/keyrings/eduvpn-v4.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/eduvpn-v4.gpg] https://app.eduvpn.org/linux/v4/deb/ noble main" | sudo tee /etc/apt/sources.list.d/eduvpn-v4.list
sudo apt update
sudo apt install eduvpn-client
```

## Tailscale
- Needs curl
- tailscale