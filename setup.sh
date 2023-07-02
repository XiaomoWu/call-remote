#!/bin/bash

ip="104.171.202.107"

# ssh to remote server
# ssh -i ~/.ssh/lambda.pem ubuntu@

# copy data
rsync -hrtv /home/yu/OneDrive/Call/remote-dev/data -e "ssh -i .ssh/lambda.pem" ubuntu@${ip}:~/call-remote

# system update
sudo apt-get update && sudo apt-get dist-upgrade 

# install python
# sudo apt install python3.11-full
# sudo apt install python3.11-dev
# sudo apt install python3.11-venv
# sudo apt install python3-pip


# install packages
python3 -m pip install --upgrade pip
pip3 install -U wandb deepspeed python-dotenv hydra-core lightning pyarrow transformers

# run unit-test
cd ~/Call/call-remote
python code/v2/model/run.py experiment=unit-test

# run wandb sweep
wandb sweep /Call/call-remote/code/v2/model/configs/experiment/swp-frtxt.yaml

