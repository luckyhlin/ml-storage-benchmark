#!/bin/bash

sudo rm -f profile_output.log && python3 src/test.py & PID=$! && sudo strace -tt -e trace=read,write,open,openat,close -p $PID -o profile_output.log
