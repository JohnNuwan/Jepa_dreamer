#!/bin/bash
cd /home/aza/ftmo_agent
source venv/bin/activate
exec python3 -u server.py checkpoints/best_model.pt 9999
