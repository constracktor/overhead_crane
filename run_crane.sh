#!/bin/bash
# create python enviroment
python3 -m venv crane_env
# activate enviroment
source crane_env/bin/activate
# install requirements
pip install --no-cache-dir -r requirements.txt
# launch game
python3 crane.py
