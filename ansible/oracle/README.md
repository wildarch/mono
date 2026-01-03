# Additional Setup for Oracle

## Security list
By default only SSH is enabled.
This helps make ports 80 and 443 reachable.

You need to enable the ports in the security list as detailed
[here](https://docs.oracle.com/en/learn/lab_compute_instance/index.html#connect-to-the-instance-and-install-apache-http-server).

## Install OCI client
Use the standard instructions for linux: https://github.com/oracle/oci-cli?tab=readme-ov-file#linux

Then do:

```bash
oci setup config
```

Generate a key pair and upload the public key to the oracle profile.
Wait a few minutes for this to propagate, then test with:

```bash
# Initially fails with 'NotAuthenticated', succeeds after the public key has
# been registered.
oci os ns get
```
