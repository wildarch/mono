Write a python script `provision/provision.py` to provision a fresh install, installing additional packages and applying system configuration, like the ansible configuration under `ansible/`. 

The tool runs one or more 'tasks'.
Each task is a python function:

```python
def install_spotify(ctx):
    # if spotify is already installed, return early
    ...

    # download and install the spotify package
    ...

register_task('install_spotify', install_spotify)
```

Tasks are registered using the `register_task` function.
It should be easy to put different tasks into separate files.
Tasks are idempotent: they detect whether any changes are necessary, and return early if not.

The main `provision.py` script includes default collections of tasks to run, for example `--system thinkpad` will run all tasks to provision the Thinkpad laptop. Alternatively, the flag `--task <TASK NAME>` can be used to run a single task in isolation.
Some tasks may need to be run as root, others as the current user.
For this the `provision.py` script includes `--root` and `--user` flags.
Correct usage of the flags is asserted by checking the UID (must be 0 for `--root`, >0 for `--user`).
If neither `--root` nor `--user` is given, the UID under which the script runs is used to select the mode.