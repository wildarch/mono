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

$ 4. Restore personal configuration
bazel run //configsync restore

# 5. Once SSH Keys are setup, change to an SSH url:
git remote set-url origin git@github.com:wildarch/mono.git

# 6. Open in VS Code, and click re-open in container button.
```

# TODO move to devcontainer
- ansible
- columnar read parquet
