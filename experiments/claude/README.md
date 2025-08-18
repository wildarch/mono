# Setup Claude in Dev Container
```bash
sudo apt update
sudo apt install npm --no-install-recommends

# Setup NPM prefix
mkdir -p ~/.local
npm config set prefix ~/.local
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install Claude Code
npm install -g @anthropic-ai/claude-code
```
