# Setting up Gemini in dev container

```bash
sudo apt update
sudo apt install -y curl

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

sudo apt install -y nodejs

# Setup NPM prefix
mkdir -p ~/.local
npm config --location user set prefix ~/.local
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install Gemini CLI
npm install -g @google/gemini-cli
```
