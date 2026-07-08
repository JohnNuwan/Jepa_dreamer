#!/bin/bash
cd /home/aza/ftmo_agent
source venv/bin/activate
exec python3 -u train_v2.py 1000
