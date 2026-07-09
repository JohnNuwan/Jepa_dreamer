"""
ftmo_agent/server.py — Live inference server.
Receives market data from local PC via TCP, runs model, sends back signals.
"""
import socket
import json
import threading
import pickle
import numpy as np
import torch
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import compute_features
from agent import PPOTrainer, TradingActorCritic

HOST = '0.0.0.0'
PORT = 9999

class InferenceServer:
    """Server that receives market data and returns trading signals."""
    
    def __init__(self, model_path='checkpoints/best_model.pt', port=9999):
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.n_features = 38  # 35 features + 3 position info
        self.model = TradingActorCritic(self.n_features, n_actions=4).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"WARNING: No model at {model_path}, using random weights")
        
        self.model.eval()
        
        # State
        self.current_position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.position_bars = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.lot_size = 0
        self.balance = 10000
        self.peak_balance = 10000
        self.daily_start = 10000
        self.trades_today = 0
        
        print(f"Inference server ready on port {port}")
        print(f"Device: {self.device}")
    
    def process_bar(self, bars_df):
        """
        Process a new bar and return trading signal.
        bars_df: DataFrame with last 200 OHLCV bars
        Returns: dict with action and parameters
        """
        # Compute features
        df, feature_cols = compute_features(bars_df, lookback=48)
        
        if len(df) < 48:
            return {'action': 'HOLD', 'reason': 'insufficient_data'}
        
        # Get last 48 bars of features
        feat = df[feature_cols].iloc[-48:].values
        
        # Add position info
        pos_info = np.array([[
            self.current_position,
            self.position_bars / 96.0,
            (self.entry_price - df.iloc[-1]['close']) / (df.iloc[-1]['close'] + 1e-8) 
            if self.current_position != 0 else 0
        ]])
        pos_repeated = np.tile(pos_info, (48, 1))
        obs = np.hstack([feat, pos_repeated]).astype(np.float32)
        
        # Model prediction
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            action = logits.argmax(dim=-1).item()
            probs = torch.softmax(logits, dim=-1).squeeze()
        
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
        
        # Check if we should close existing position
        if self.current_position != 0:
            # Check SL/TP
            current_price = df.iloc[-1]['close']
            atr = df.iloc[-1].get('atr_norm', 0.01) * current_price
            
            if self.current_position == 1:  # long
                if current_price <= self.stop_loss:
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'SL', 'price': current_price}
                elif current_price >= self.take_profit:
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'TP', 'price': current_price}
                elif action == 3:  # model says close
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'MODEL', 'price': current_price}
            elif self.current_position == -1:  # short
                if current_price >= self.stop_loss:
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'SL', 'price': current_price}
                elif current_price <= self.take_profit:
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'TP', 'price': current_price}
                elif action == 3:
                    self._close_position(current_price)
                    return {'action': 'CLOSE', 'reason': 'MODEL', 'price': current_price}
        
        # Check if we should open new position
        if self.current_position == 0 and action in [1, 2]:
            if self.trades_today < 5:
                current_price = df.iloc[-1]['close']
                atr = max(df.iloc[-1].get('atr_norm', 0.01) * current_price, 2.0)
                spread = 0.02  # $0.02 spread on XAUUSD
                
                if action == 1:  # BUY
                    entry = current_price + spread / 2
                    sl = entry - atr * 1.5
                    tp = entry + atr * 3.0
                    self.current_position = 1
                else:  # SELL
                    entry = current_price - spread / 2
                    sl = entry + atr * 1.5
                    tp = entry - atr * 3.0
                    self.current_position = -1
                
                self.entry_price = entry
                self.stop_loss = sl
                self.take_profit = tp
                self.position_bars = 0
                
                # Position sizing: 1% risk
                risk_amount = self.balance * 0.01
                sl_distance = abs(entry - sl)
                pip_value = 100  # $100 per $1 move per lot
                self.lot_size = max(0.01, min(risk_amount / (sl_distance * pip_value), 0.5))
                self.trades_today += 1
                
                return {
                    'action': 'BUY' if action == 1 else 'SELL',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'lots': self.lot_size,
                    'probabilities': probs.tolist(),
                    'value': value.item(),
                }
        
        self.position_bars += 1 if self.current_position != 0 else 0
        return {
            'action': 'HOLD',
            'probabilities': probs.tolist(),
            'value': value.item(),
        }
    
    def _close_position(self, exit_price):
        """Close position and update balance."""
        if self.current_position == 1:
            pnl = (exit_price - self.entry_price) * self.lot_size * 100
        else:
            pnl = (self.entry_price - exit_price) * self.lot_size * 100
        
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        self.current_position = 0
        self.entry_price = 0
        self.position_bars = 0
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_start = self.balance
        self.trades_today = 0
    
    def get_status(self):
        return {
            'position': {0: 'FLAT', 1: 'LONG', -1: 'SHORT'}[self.current_position],
            'balance': self.balance,
            'entry_price': self.entry_price,
            'sl': self.stop_loss,
            'tp': self.take_profit,
            'lots': self.lot_size,
            'trades_today': self.trades_today,
        }
    
    def run(self):
        """Start TCP server."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, self.port))
        server.listen(1)
        print(f"Server listening on {HOST}:{self.port}")
        
        while True:
            try:
                client, addr = server.accept()
                print(f"Client connected: {addr}")
                handler = ClientHandler(client, self)
                handler.start()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Server error: {e}")

class ClientHandler(threading.Thread):
    def __init__(self, client, server):
        super().__init__(daemon=True)
        self.client = client
        self.server = server
    
    def run(self):
        buffer = ""
        while True:
            try:
                data = self.client.recv(65536).decode('utf-8')
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    msg = json.loads(line)
                    
                    if msg['type'] == 'bars':
                        # Process bars
                        import pandas as pd
                        df = pd.DataFrame(msg['data'])
                        signal = self.server.process_bar(df)
                        
                        response = json.dumps({
                            'type': 'signal',
                            'signal': signal,
                            'status': self.server.get_status(),
                            'timestamp': datetime.now().isoformat(),
                        }) + '\n'
                        self.client.send(response.encode('utf-8'))
                    
                    elif msg['type'] == 'status':
                        response = json.dumps({
                            'type': 'status',
                            'status': self.server.get_status(),
                        }) + '\n'
                        self.client.send(response.encode('utf-8'))
                    
                    elif msg['type'] == 'reset_daily':
                        self.server.reset_daily()
                        response = json.dumps({'type': 'ack', 'msg': 'daily reset'}) + '\n'
                        self.client.send(response.encode('utf-8'))
                    
                    elif msg['type'] == 'trade_closed':
                        # Update server state when MT5 confirms a trade closed
                        exit_price = msg.get('exit_price', 0)
                        self.server._close_position(exit_price)
                        if msg.get('daily_reset'):
                            self.server.reset_daily()
                        response = json.dumps({'type': 'ack'}) + '\n'
                        self.client.send(response.encode('utf-8'))
                        
            except Exception as e:
                print(f"Handler error: {e}")
                break
        
        self.client.close()

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_model.pt'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    server = InferenceServer(model_path=model_path, port=port)
    server.run()
