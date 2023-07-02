#!/bin/bash

ip="104.171.202.107"

# ssh to remote server
# ssh -i ~/.ssh/lambda.pem ubuntu@

# copy data
rsync -hrtv /home/yu/OneDrive/Call/remote-dev/data -e "ssh -i .ssh/lambda.pem" ubuntu@${ip}:~/call-remote

# system update
sudo apt-get update && sudo apt-get dist-upgrade 


# install packages
python3 -m pip install --upgrade pip
pip3 install -U wandb deepspeed python-dotenv hydra-core lightning pyarrow transformers numexpr
pip3 install protobuf==3.20  # v4.0 is incompatible with wandb

# init wandb
pip install -U click  # required for wandb, see https://community.wandb.ai/t/unable-to-login/4335/15?u=yuzhu


# run unit-test
cd ~/Call/call-remote
python code/v2/model/run.py experiment=unit-test

# run wandb sweep
wandb sweep ~/Call/call-remote/code/v2/model/configs/experiment/swp-frtxt.yaml

# download data from lambda to local
# rsync -hrtv ubuntu@${ip}:~/call-remote/data ~/OneDrive/Call/remote-dev