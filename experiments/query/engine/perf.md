# Using perf
1. Install perf for your kernel version. I used `sudo apt install linux-tools-5.19.0-50-generic`.
2. Allow data collection: `sudo sysctl -w kernel.perf_event_paranoid=2`
2. Collect some data: `perf record --call-graph dwarf <your application>`
3. Analyse in hotspot: `hotspot perf.data`