# FTMO Agent — Evolution Strategies pour Challenges Prop Firm

**Agent de trading entraîné par Evolution Strategies (ES) à passer les challenges FTMO** sur XAUUSD, EURUSD, indices US/EU et BTCUSD en M15, avec LSTM + curriculum learning + antithetic sampling corrigé sur 2 GPUs.

## Architecture

```
ftmo_agent/
├── train_es.py           ← Point d'entrée : entraînement ES V5
├── es_agent.py           ← Agent ES : LSTM 2×128, antithetic corrigé, dual GPU
├── config.py             ← Configuration : symboles, spreads, règles FTMO, curriculum
├── environment.py        ← Environnement V4 : spread variable, slippage, commission, PnL pur
├── features_v2.py        ← Features multi-timeframes (M15/H1/H4/D1)
├── test_synthetic.py     ← Tests synthétiques de l'agent
├── checkpoints_es/       ← Modèles ES sauvegardés
├── data/                 ← Données OHLC M15 par symbole
└── legacy/               ← Versions archivées (DreamerV3, PPO, V1-V3)
```

## Evolution Strategies V5

L'agent utilise une **Evolution Strategies** avec les caractéristiques suivantes :

### Politique LSTM 2×128

- **Réseau** : LSTM à 2 couches cachées de 128 neurones chacune
- **Entrée** : ~296 features multi-timeframes (M15, H1, H4, D1) × 48 barres de lookback
- **Sortie** : 8 logits d'action (HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE)
- **~368K paramètres**

### Bias anti-HOLD + frozen_action_mask (V5)

Un buffer fixe **non-apprenable** + un masque de gel garantissent que HOLD/BUY/SELL sont déterminés uniquement par le bias, jamais par le réseau :

| Action | Bias | Canal réseau | Rôle |
|--------|------|-------------|------|
| HOLD | -4.0 | Gelé | Jamais choisi sans position |
| BUY | +3.0 | Gelé | Toujours favorisé |
| SELL | +3.0 | Gelé | Toujours favorisé |
| CLOSE | +0.5 | Appris | Le réseau apprend QUAND fermer |
| SPLIT_BUY | +1.5 | Appris | Le réseau apprend le sizing |
| SPLIT_SELL | +1.5 | Appris | Le réseau apprend le sizing |
| PYRAMID | 0.0 | Appris | Renforcement conditionnel |
| PARTIAL_CLOSE | 0.0 | Appris | Prise de profit partielle |

**Pourquoi geler HOLD/BUY/SELL** : l'agent n'a pas besoin d'apprendre la direction — le timing de sortie et le sizing sont les vraies compétences à acquérir.

### Antithetic sampling corrigé (V5)

Pour chaque individu de la population, on évalue **deux politiques opposées** sur le **MÊME marché** (même symbole, même step de départ) :
- `master + noise` (fitness_plus)
- `master - noise` (fitness_minus)

La **fitness effective** = `fitness_plus - fitness_minus` isole l'effet de la perturbation ε du bruit du marché.

> **BUGFIX V5** : Avant, -ε était évalué sur un marché DIFFÉRENT (reset() aléatoire) → gradient = bruit pur.

### Dual GPU

L'évaluation de la population est parallélisée sur **2 GPUs** (`cuda:0` et `cuda:1`) via `ThreadPoolExecutor`.

### Fitness = PnL pur

```
fitness = (balance_final - balance_initial) / balance_initial × 100
```

- **Pénalité zéro-trade** : -50 si aucun trade
- **Bonus trading** : +2 si au moins un trade

### Température stochastique décroissante

- **Début** : `temp = 1.5` (forte exploration)
- **Fin** : `temp = 0.3` (exploitation)
- **Décroissance** : sur 150 générations

### Sélection par élite

- Top 25% de la population
- Pondération par rang
- Gradient normalisé

## Curriculum 3 phases

| Phase | Épisodes | Spread | Slippage | Commission | Trades max |
|-------|----------|--------|----------|------------|------------|
| **Phase 1** | 0–200 | 0% | 0% | 0% | 20 |
| **Phase 2** | 200–500 | 30% | 30% | 0% | 12 |
| **Phase 3** | 500+ | 100% | 100% | 100% | 8 |

## Utilisation

```bash
# Entraînement
cd ftmo_agent
python3 train_es.py

# Tests synthétiques
python3 test_synthetic.py
```

## Configuration

Les paramètres principaux sont dans `config.py` :

- **7 symboles actifs** : XAUUSD, EURUSD, GBPUSD, US30, GER40, US500, US100, BTCUSD
- **8 actions** : HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE
- **Règles FTMO** : DD quotidien 5%, DD total 10%, profit target 10%, max 8 trades/jour
- **Risk** : 0.5% par trade, SL à 2 ATR, TP à 4 ATR
- **ES V5** : population 32, σ=0.02, lr=0.1, 25% élite, antithetic corrigé
- **Bias fixe** : HOLD=-4.0, BUY=+3.0, SELL=+3.0, canaux gelés via frozen_action_mask

## Historique des versions

| Version | Algorithme | Statut | Raison de l'échec |
|---------|-----------|--------|-------------------|
| V1-V3 | DreamerV3 (15.5M params) | ❌ Abandonné | reward_head collapse vers la moyenne |
| V4.0 | PPO | ❌ Abandonné | Entropie → 0, pas de trades |
| V4.1 | ES (pop=8) | ⚠️ Insuffisant | Apprend BUY/SELL sur synthétique, pas sur réel |
| V4.2 | ES V4 (pop=16, antithetic) | ⚠️ Stagne | Antithetic cassé (marchés différents), gradient = bruit |
| **V5** | **ES V5 (pop=32, antithetic corrigé, bias gelé)** | 🔄 En cours | Antithetic fixé, HOLD/BUY/SELL gelés, réseau apprend le timing |
