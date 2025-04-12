# mono
My own personal monorepo.

# Bootstrapping
```
# 1. Download git
sudo apt install git

# 2. Clone the repository
git clone https://github.com/wildarch/mono.git

# 3. Run commands below
sudo apt install ansible
ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/zbook.yml -K

# 4. Install dependencies for building the monorepo
ansible-playbook ansible/build.yml -K

$ 5. Restore personal configuration
bazel run //configsync restore

# 6. Once SSH Keys are setup, change to an SSH url:
git remote set-url origin git@github.com:wildarch/mono.git
```

# TODO move to devcontainer
- configsync
- ansible
- GitHub CI
- columnar lit tests
- columnar read parquet
