# FTMO Agent — Evolution Strategies pour Challenges Prop Firm

**Agent de trading entraîné par Evolution Strategies (ES) à passer les challenges FTMO** sur XAUUSD, EURUSD, indices US/EU et BTCUSD en M15, avec LSTM + curriculum learning + antithetic sampling sur 2 GPUs.

## Architecture

```
ftmo_agent/
├── train_es.py           ← Point d'entrée : entraînement ES avec curriculum
├── es_agent.py           ← Agent ES : LSTM 2×128, antithetic, dual GPU
├── config.py             ← Configuration : symboles, spreads, règles FTMO, curriculum
├── environment.py        ← Environnement V4 : spread variable, slippage, commission, PnL pur
├── features_v2.py        ← Features multi-timeframes (M15/H1/H4/D1)
├── test_synthetic.py     ← Tests synthétiques de l'agent
├── checkpoints_es/       ← Modèles ES sauvegardés
├── data/                 ← Données OHLC M15 par symbole
└── legacy/               ← Versions archivées (DreamerV3, PPO, V1-V3)
```

## Evolution Strategies (ES)

L'agent utilise une **Evolution Strategies** avec les caractéristiques suivantes :

### Politique LSTM 2×128

- **Réseau** : LSTM à 2 couches cachées de 128 neurones chacune
- **Entrée** : ~296 features multi-timeframes (M15, H1, H4, D1) × 48 barres de lookback
- **Sortie** : 8 logits d'action (HOLD, BUY, SELL, CLOSE, SPLIT_BUY, SPLIT_SELL, PYRAMID, PARTIAL_CLOSE)
- **~500K paramètres** apprenables

### Bias anti-HOLD

Un buffer fixe **non-apprenable** (exclu des paramètres ES) est ajouté aux logits pour contrer la tendance naturelle à ne jamais trader :

| Action | Bias |
|--------|------|
| HOLD | -2.0 |
| BUY | +1.0 |
| SELL | +1.0 |
| SPLIT_BUY | +0.5 |
| SPLIT_SELL | +0.5 |
| CLOSE, PYRAMID, PARTIAL_CLOSE | 0.0 |

### Antithetic sampling

Pour chaque individu de la population, on évalue **deux politiques opposées** :
- `master + noise` (fitness_plus)
- `master - noise` (fitness_minus)

La **fitness effective** = `fitness_plus - fitness_minus` donne un gradient **2× plus précis** et réduit la variance.

### Dual GPU

L'évaluation de la population est parallélisée sur **2 GPUs** (`cuda:0` et `cuda:1`) via `ThreadPoolExecutor`. Chaque politique est assignée à un GPU en round-robin, et les évaluations antithetic sont dispatchées sur les deux GPUs simultanément.

### Fitness = PnL pur

La fitness est basée uniquement sur le **PnL réalisé** :

```
fitness = (balance_final - balance_initial) / balance_initial × 100
```

- **Pénalité zéro-trade** : -50 si aucun trade n'est pris pendant l'évaluation
- **Bonus trading** : +2 si au moins un trade est effectué
- Aucun reward shaping dense — l'agent est jugé sur son résultat final

### Température stochastique décroissante

L'exploration est contrôlée par une température qui décroît linéairement :
- **Début** : `temp = 1.5` (forte exploration, sampling softmax)
- **Fin** : `temp = 0.3` (exploitation, quasi-greedy)
- **Décroissance** : sur 150 générations

### Sélection par élite

- **Top 25%** de la population conservé comme élite
- Pondération par rang (ranking-based weights)
- Gradient normalisé par le nombre d'élites et la fitness signée

## Curriculum 3 phases

L'environnement augmente progressivement la difficulté :

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
- **ES** : population 16, σ=0.02, lr=0.1, 25% élite

## Résultats

L'approche ES avec antithetic sampling et curriculum learning permet à l'agent d'apprendre sans backpropagation, en optimisant directement le PnL réalisé via des perturbations aléatoires de la politique.