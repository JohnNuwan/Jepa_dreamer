# FTMO Agent — DreamerV3 for Prop Firm Challenges

**RL agent entraîné à passer les challenges FTMO** en tradant XAUUSD, EURUSD, indices US/EU et BTCUSD sur M15, avec un DreamerV3 (world model + actor-critic + JEPA) sur 2 GPUs.

## Architecture

```
ftmo_agent/
├── train.py              ← Point d'entrée : entraînement V4 avec curriculum
├── diag.py               ← Diagnostic : charger un modèle et analyser les actions
├── config.py             ← Configuration : symboles, spreads, FTMO rules, curriculum
├── environment.py        ← Environnement V4 : spread variable, slippage, commission, pure PnL
├── features_v2.py        ← Features V4 : 69/TF multi-timeframes (M15/H1/H4/D1)
├── dreamer_trainer_v2.py ← DreamerV3 : 15.5M params sur 2 GPUs (JEPA + RSSM + ActorCritic)
├── octopus/              ← Core ML : world model, JEPA self-supervised, actor-critic
├── data/                 ← Données OHLC M15 par symbole
├── legacy/               ← Anciennes versions (V1-V3) archivées
└── checkpoints_v4/       ← Modèles sauvegardés
```

## Changements V4

| Aspect | V3 (avant) | V4 (maintenant) |
|--------|-----------|-----------------|
| **Reward** | Shaping (pénalités, bonus) | **Pure PnL** (% du compte) |
| **Spread** | Fixe | Variable (session × volatilité × aléa) |
| **Slippage** | Aucun | Gaussien (~30% du spread) |
| **Commission** | Aucune | $7/lot MT5 standard |
| **Curriculum** | Non | 3 phases : 0% → 30% → 100% frictions |
| **Features** | 33/TF | **69/TF** : HV, slopes, lag, corrélations |
| **TP/SL** | 1.5/3.0 ATR | 2.0/4.0 ATR |
| **Risque** | 1%/trade | 0.5%/trade |

## Utilisation

```bash
# Entraînement
cd ftmo_agent
python3 train.py --episodes 3000

# Diagnostic
python3 diag.py
```

## Résultats attendus

Le curriculum V4 permet à l'agent d'apprendre progressivement :
1. **Phase 1** (0-200ep) : spread nul, pas de slippage → apprendre à trader
2. **Phase 2** (200-500ep) : spread 30%, slippage réduit → affiner la sélection
3. **Phase 3** (500+ ep) : spread réel, slippage, commission → environnement réaliste FTMO