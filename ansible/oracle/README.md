# Allowing HTTP Access to Oracle Compute
By default only SSH is enabled.
This helps make ports 80 and 443 reachable.

## Firewall
*Note: The Ansible playbook should take care of this.*

Follow the guide 
[here](https://blogs.oracle.com/developers/post/enabling-network-traffic-to-ubuntu-images-in-oracle-cloud-infrastructure).

In essence, right after the rule for SSH (look for port 22), you need to add two lines:

```
# In /etc/iptables/rules.v4
-A INPUT -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT
-A INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT
```

Then reload firewall settings with `sudo iptables-restore < /etc/iptables/rules.v4`

## Security list
You also need to enable the ports in the security list as detailed 
[here](https://docs.oracle.com/en/learn/lab_compute_instance/index.html#connect-to-the-instance-and-install-apache-http-server).