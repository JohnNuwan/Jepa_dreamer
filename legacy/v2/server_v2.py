"""
ftmo_agent/server_v2.py — Multi-symbol live inference server.
Receives bar data for any symbol, returns signal with symbol-aware parameters.
"""
import socket, json, threading, numpy as np, torch, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, ACTIVE_SYMBOLS, FTMO_CONFIG, RISK_CONFIG,
                     N_ACTIONS, ACTION_NAMES, BUY, SELL, CLOSE, HOLD,
                     SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE)
from features_v2 import compute_multi_tf_features, get_symbol_embedding
from agent_v2 import MultiSymbolActorCritic

HOST = '0.0.0.0'
PORT = 9999

class MultiSymbolServer:
    def __init__(self, model_path='checkpoints_v2/best_model.pt', port=9999):
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature count: 32 (M15) + 32 (H1) + 32 (H4) + 32 (D1) + 8 (symbol) + 5 (position) = 141
        self.n_features = 128 + 8 + 5  # 141
        self.model = MultiSymbolActorCritic(self.n_features, n_actions=N_ACTIONS).to(self.device)
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            print(f"Model loaded: {model_path}")
        else:
            print(f"WARNING: No model at {model_path}")
        self.model.eval()
        
        # State per symbol
        self.positions = {}  # symbol → list of position dicts
        for s in ACTIVE_SYMBOLS:
            self.positions[s] = []
        
        self.balance = FTMO_CONFIG['account_size']
        self.peak_balance = self.balance
        self.daily_start = self.balance
        self.trades_today = 0
        
        print(f"Server ready on port {port}, device={self.device}")
    
    def process_bar(self, symbol, bars_df):
        """Process new bar and return signal."""
        if symbol not in SYMBOLS:
            return {'action': 'HOLD', 'reason': 'unknown_symbol'}
        
        spec = SYMBOLS[symbol]
        
        # Compute multi-TF features
        features, feat_names, df = compute_multi_tf_features(bars_df, lookback=48)
        
        if len(features) < 48:
            return {'action': 'HOLD', 'reason': 'insufficient_data'}
        
        # Get last 48 bars
        feat = features[-48:]
        
        # Symbol embedding
        sym_emb = get_symbol_embedding(symbol)
        sym_tiled = np.tile(sym_emb, (48, 1))
        
        # Position info
        positions = self.positions.get(symbol, [])
        current_price = df.iloc[-1]['close']
        n_pos = len(positions)
        total_unreal = sum(p.get('unrealized', 0) for p in positions)
        net_dir = sum(p.get('direction', 0) * p.get('lots', 0) for p in positions)
        total_lots = sum(p.get('lots', 0) for p in positions)
        avg_bars = np.mean([p.get('bars_held', 0) for p in positions]) if positions else 0
        
        pos_info = np.array([[
            n_pos / 3.0, total_unreal / self.balance,
            net_dir / (total_lots + 1e-8), total_lots / spec.max_volume,
            avg_bars / FTMO_CONFIG['max_hold_bars'],
        ]])
        pos_tiled = np.tile(pos_info, (48, 1))
        
        obs = np.hstack([feat, sym_tiled, pos_tiled]).astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            action = logits.argmax(dim=-1).item()
        
        # Check existing positions (SL/TP/SLBE)
        if positions:
            for i in range(len(positions) - 1, -1, -1):
                p = positions[i]
                p['bars_held'] = p.get('bars_held', 0) + 1
                
                # SLBE check
                unreal = self._calc_unrealized(p, current_price, spec)
                p['unrealized'] = unreal
                if unreal >= RISK_CONFIG['slbe_trigger_usd'] and not p.get('slbe'):
                    p['slbe'] = True
                    return {'action': 'MODIFY_SL', 'symbol': symbol, 'ticket': p.get('ticket'),
                            'new_sl': p['entry'] + (RISK_CONFIG['slbe_offset_usd'] / (p['lots'] * spec.contract_size) * p['direction']),
                            'reason': 'SLBE'}
                
                # SL/TP check
                if p['direction'] == 1:
                    if current_price <= p['sl']:
                        return {'action': 'CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'), 'reason': 'SL'}
                    if current_price >= p['tp']:
                        return {'action': 'CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'), 'reason': 'TP'}
                else:
                    if current_price >= p['sl']:
                        return {'action': 'CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'), 'reason': 'SL'}
                    if current_price <= p['tp']:
                        return {'action': 'CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'), 'reason': 'TP'}
                
                # Model says close
                if action == CLOSE:
                    return {'action': 'CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'), 'reason': 'MODEL'}
                
                # Partial close
                if action == PARTIAL_CLOSE and unreal > 0 and not p.get('partial'):
                    p['partial'] = True
                    return {'action': 'PARTIAL_CLOSE', 'symbol': symbol, 'ticket': p.get('ticket'),
                            'close_lots': p['lots'] * RISK_CONFIG['partial_close_pct'], 'reason': 'PARTIAL'}
        
        # Open new position
        if not positions and action in [BUY, SELL, SPLIT_BUY, SPLIT_SELL]:
            if self.trades_today < FTMO_CONFIG['max_trades_per_day']:
                spread = spec.spread_points * spec.pip_size
                atr_norm_idx = feat_names.index('atr_norm') if 'atr_norm' in feat_names else -1
                atr_val = abs(features[-1, atr_norm_idx]) if atr_norm_idx >= 0 else spec.volatility_pct
                atr = max(atr_val * current_price, spec.pip_size * 10)
                
                if action in [BUY, SPLIT_BUY]:
                    direction = 1
                    entry = current_price + spread / 2
                else:
                    direction = -1
                    entry = current_price - spread / 2
                
                sl = entry - direction * atr * 1.5
                tp = entry + direction * atr * 3.0
                
                # Position sizing
                risk = RISK_CONFIG['risk_per_trade']
                if action in [SPLIT_BUY, SPLIT_SELL]:
                    risk *= 0.5  # half risk for split
                risk_amount = self.balance * risk
                sl_dist = abs(entry - sl)
                lots = max(spec.min_volume, min(risk_amount / (sl_dist * spec.contract_size), spec.max_volume))
                
                self.trades_today += 1
                return {
                    'action': ACTION_NAMES[action],
                    'symbol': symbol, 'direction': 'BUY' if direction == 1 else 'SELL',
                    'entry': entry, 'sl': sl, 'tp': tp, 'lots': lots,
                    'split': action in [SPLIT_BUY, SPLIT_SELL],
                    'probabilities': probs.tolist(), 'value': value.item(),
                }
        
        return {
            'action': 'HOLD', 'symbol': symbol,
            'probabilities': probs.tolist(), 'value': value.item(),
        }
    
    def _calc_unrealized(self, pos, price, spec):
        return (price - pos['entry']) * pos['lots'] * spec.contract_size * pos['direction']
    
    def on_trade_opened(self, symbol, ticket, direction, entry, lots, sl, tp):
        """Called when MT5 confirms order."""
        self.positions.setdefault(symbol, []).append({
            'ticket': ticket, 'direction': direction, 'entry': entry,
            'lots': lots, 'sl': sl, 'tp': tp, 'bars_held': 0,
            'slbe': False, 'partial': False,
        })
    
    def on_trade_closed(self, symbol, ticket, pnl):
        """Called when MT5 confirms close."""
        if symbol in self.positions:
            self.positions[symbol] = [p for p in self.positions[symbol] if p.get('ticket') != ticket]
        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
    
    def reset_daily(self):
        self.daily_start = self.balance
        self.trades_today = 0
    
    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, self.port))
        server.listen(5)
        print(f"Listening on {HOST}:{self.port}")
        
        while True:
            try:
                client, addr = server.accept()
                handler = threading.Thread(target=self._handle, args=(client,), daemon=True)
                handler.start()
            except KeyboardInterrupt:
                break
    
    def _handle(self, client):
        buf = ""
        while True:
            try:
                data = client.recv(65536).decode('utf-8')
                if not data: break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    msg = json.loads(line)
                    resp = self._process_msg(msg)
                    client.send((json.dumps(resp) + '\n').encode())
            except Exception as e:
                print(f"Handler: {e}")
                break
        client.close()
    
    def _process_msg(self, msg):
        if msg['type'] == 'bars':
            import pandas as pd
            df = pd.DataFrame(msg['data'])
            df['time'] = pd.to_datetime(df['time'])
            signal = self.process_bar(msg['symbol'], df)
            return {'type': 'signal', 'signal': signal, 'timestamp': datetime.now().isoformat()}
        elif msg['type'] == 'trade_opened':
            self.on_trade_opened(msg['symbol'], msg['ticket'], msg['direction'],
                                 msg['entry'], msg['lots'], msg['sl'], msg['tp'])
            return {'type': 'ack'}
        elif msg['type'] == 'trade_closed':
            self.on_trade_closed(msg['symbol'], msg['ticket'], msg.get('pnl', 0))
            return {'type': 'ack'}
        elif msg['type'] == 'reset_daily':
            self.reset_daily()
            return {'type': 'ack'}
        elif msg['type'] == 'status':
            return {'type': 'status', 'balance': self.balance,
                    'positions': {s: len(p) for s, p in self.positions.items()}}
        return {'type': 'error', 'msg': 'unknown type'}

if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_v2/best_model.pt'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    MultiSymbolServer(model, port).run()
