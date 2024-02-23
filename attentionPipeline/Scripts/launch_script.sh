#!/bin/bash

echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '\''{print $2}'\''):0' >> ~/.bashrc
source ~/.bashrc
export DISPLAY=127.0.0.1:0