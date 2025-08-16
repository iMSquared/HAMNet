#!/usr/bin/env bash

cd ~
cp -r /opt/isaacgym ~/
python3 -m pip install -e ~/isaacgym/python
cd /tmp && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip && rm -f eigen-3.4.0.zip

# TODO change name of the repo
python3 -m pip install --no-build-isolation -e ~/HAMNet/ham
git config --global --add safe.directory ~/HAMNet
sudo chmod 777 -R /home/user/.cache
