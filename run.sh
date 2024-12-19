#!/bin/bash

# do not run bash file directly, but copy and paste the following command to CMDLine instead.
sudo rm -f profile_output.log && python3 src/test.py & PID=$! && sudo strace -tt -e trace=read,write,open,openat,close -p $PID -o profile_output.log && sudo chown $USER:$USER profile_output.log
