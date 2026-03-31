# regime-rl-trading

**Regime-aware Reinforcement Learning for Adaptive Trading Strategy Selection**

A modular Python framework that detects market regimes (Bull / Bear / Sideways / Volatile)
and uses RL agents to dynamically allocate capital across regime-optimised trading strategies.

---

## Architecture

```
Market Data → FeatureEngineer → RegimeDetector ──┐
                                                  ↓
                                        TradingEnv (Gymnasium)
                                                  ↓
                                         RL Agent (PPO / DQN / Meta)
                                                  ↓
                                Strategy Blending (4 strategies)
                                                  ↓
                                          Backtester → Metrics + Plots
```

### Regime Detection
| Detector | Description |
|---|---|
| `FeatureRegimeDetector` | Rule-based thresholds on volatility, trend & momentum (no training required) |
| `HMMRegimeDetector` | Gaussian HMM with automatic state → regime mapping via `hmmlearn` |

### Strategies
| Strategy | Regime Fit | Core Signal |
|---|---|---|
| `MomentumStrategy` | BULL | Short/long MA crossover + 10-day momentum |
| `MeanReversionStrategy` | SIDEWAYS | RSI + Bollinger Band extremes |
| `BreakoutStrategy` | VOLATILE | Elevated ATR near recent high/low |
| `DefensiveStrategy` | BEAR | Near-zero exposure during high volatility |

### Agents
| Agent | Description |
|---|---|
| `PPOAgent` | Stable-Baselines3 PPO with `MlpPolicy` |
| `DQNAgent` | Stable-Baselines3 DQN with discrete strategy selection |
| `MetaAgent` | Ensemble of per-regime PPO agents; routes by detected regime |

---

## Project Structure

```
regime-rl-trading/
├── config/default.yaml          # All hyperparameters
├── train.py                     # Training entry-point
├── evaluate.py                  # Evaluation entry-point
├── notebooks/exploration.ipynb  # Interactive exploration
├── src/
│   ├── regime_detection/        # HMM & feature-based detectors
│   ├── strategies/              # 4 trading strategies
│   ├── environment/             # Gymnasium TradingEnv + features
│   ├── agents/                  # PPO, DQN, MetaAgent
│   └── evaluation/              # Backtester + Visualizer
└── tests/                       # pytest smoke tests
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py --config config/default.yaml --ticker SPY
# Model saved to results/model/ppo_SPY.zip
```

### 3. Evaluate

```bash
python evaluate.py --config config/default.yaml --ticker SPY
# Prints metrics; saves plots to results/plots/
```

### 4. Explore interactively

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests that require heavy dependencies (`hmmlearn`, `torch`, `stable-baselines3`) are
skipped automatically when those packages are not installed.

---

## Configuration

All settings live in `config/default.yaml`:

```yaml
agent:
  type: ppo          # ppo | dqn
  training_episodes: 1000

regime_detection:
  method: feature    # feature | hmm

data:
  tickers: [SPY]
  start_date: "2015-01-01"
  end_date:   "2023-12-31"
```

---

## Performance Metrics

The `Backtester` computes:
- Total & annualised return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown
- Win rate, Profit factor
- Buy-and-hold benchmark comparison

---

## License

MIT – see [LICENSE](LICENSE).
