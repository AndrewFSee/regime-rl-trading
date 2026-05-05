"""
Gymnasium-compatible trading environment for regime-RL strategy selection.

Observation space  : Box(-inf, inf, (lookback_window * n_obs_features + 4 + 4,))
  - lookback_window * n_obs_features : rolling feature history (14 features by default)
  - 4 portfolio features              : [cash_ratio, position, unrealised_pnl, drawdown]
  - 4 regime one-hot                  : [BULL, BEAR, SIDEWAYS, VOLATILE]

Action space       : Box(0, 1, (n_strategies + 1,))
  Actions 0..3 are strategy allocation weights (softmax-normalised).
  Action 4 is the exposure scalar: 0 → flat, 1 → full max_position.
  The strategies determine direction; exposure determines position size.

Reward             : (1 - return_weight) * DSR + return_weight * net_return
                     - exposure_penalty * (1 - |position|)
                     + trend_bonus * position * trend_sign
                     - transaction_cost * turnover

  return_weight (0-1) : fraction of reward that is raw return vs DSR.
  exposure_penalty    : per-step cost for being under-invested.
  downside_only       : if True, DSR uses only downside variance (Sortino-like).
  trend_bonus         : per-step reward for being positioned with the trend.
  benchmark_relative  : if True, DSR is computed on excess returns over buy-and-hold
                        (agent_return - market_return). Penalises under-participation
                        in bull markets and rewards crash avoidance.
  bull_benchmark      : if True, excess-return DSR is applied only on up-market days;
                        on down days the agent is rewarded on absolute returns.
  exposure_floor      : minimum effective exposure (0-1). Maps agent output from
                        [0,1] to [floor,1] so the agent is always partially invested.
  hard_selection      : if True, the agent selects ONE strategy per step (argmax)
                        instead of soft-blending all four via softmax weights.
                        Gives clearer reward attribution and allows more extreme
                        positioning.
    regime_guardrail    : if True and hard_selection is enabled, BEAR and VOLATILE
                                                regimes can only select Momentum or Defensive.
    risk_off_exposure_cap:
                                                optional max exposure applied in BEAR and VOLATILE
                                                regimes after any floor remapping. Can override the floor.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required. Install with: pip install gymnasium"
    ) from exc

from .features import FeatureEngineer, FEATURE_NAMES
from ..regime_detection.feature_detector import FeatureRegimeDetector
from ..regime_detection.base import MarketRegime
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.breakout import BreakoutStrategy
from ..strategies.trend_following import TrendFollowingStrategy


_N_FEATURES_TOTAL = 24   # total features computed by FeatureEngineer
_N_STRATEGIES = 4
_N_REGIME = 4
_N_PORTFOLIO = 4
_MOMENTUM_IDX = 0
_MEAN_REVERSION_IDX = 1
_BREAKOUT_IDX = 2
_TREND_FOLLOWING_IDX = 3
_RISK_OFF_LONG_STRATEGY_INDICES = np.array([_MOMENTUM_IDX, _BREAKOUT_IDX], dtype=np.int64)

# Default: use ALL features in the observation vector.
# The lean 6-feature subset was tested and performed worse.
DEFAULT_OBS_FEATURES = FEATURE_NAMES


class _RunningNormalizer:
    """Welford online mean/variance tracker for observation normalization.

    Supports a ``training`` flag: when False, ``update()`` is a no-op so that
    evaluation rollouts use the statistics fitted on the training data. Use
    ``state_dict()`` / ``load_state_dict()`` to persist normalizer state
    alongside the policy.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small epsilon to avoid division by zero
        self.clip = clip
        self.training = True

    def update(self, x: np.ndarray) -> None:
        if not self.training:
            return
        batch_mean = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = batch_mean - self.mean
        self.mean += delta / self.count
        self.var += delta * (batch_mean - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var / self.count + 1e-8)
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "clip": float(self.clip),
            "training": bool(self.training),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self.clip = float(state.get("clip", self.clip))
        self.training = bool(state.get("training", True))


class TradingEnv(gym.Env):
    """
    Custom trading environment for regime-aware RL strategy selection.

    Parameters
    ----------
    data:               OHLCV DataFrame (Open, High, Low, Close, Volume).
    lookback_window:    Number of past time-steps in the observation (default 20).
    initial_cash:       Starting portfolio value in dollars (default 100 000).
    transaction_cost:   Fractional cost per unit of turnover (one-way). The reward
                        deducts ``transaction_cost * |new_position - old_position|``
                        each step, so the default 0.001 corresponds to ~10 bps per
                        side / ~20 bps round-trip.
    slippage_factor:    Legacy quadratic volume-impact coefficient. At retail
                        portfolio sizes on liquid ETFs (e.g. SPY) this term is
                        ~3e-7 per round-trip and effectively zero. Prefer the
                        ``slippage_bps`` / ``slippage_vol_coef`` parameters
                        below for a realistic frictions model. Default 0.0.
    slippage_bps:       Base half-spread cost in basis points charged per unit
                        of turnover (one-way). 1 bps = 0.0001. Default 1.0
                        (~retail SPY half-spread).
    slippage_vol_coef:  Adverse-selection scaler. Adds an extra cost of
                        ``slippage_vol_coef * (sigma_t / sigma_long_run)`` bps
                        per unit turnover, where sigma is realised return std.
                        Captures the empirical fact that slippage rises during
                        volatile periods. Default 0.0.
    borrow_cost_bps_yr: Annualised cost charged on short positions, in basis
                        points per year (e.g. 50 = 0.5 %% / yr). Applied per
                        step as ``-borrow_cost_bps_yr/1e4/252 * max(-pos, 0)``.
                        Default 0.0.
    reward_scaling:     Scalar multiplier applied to the reward (default 1.0).
    max_position:       Maximum absolute position fraction (default 1.0).
    normalize_obs:      Whether to apply running normalization to observations (default True).
    return_weight:      Blend weight: 0 = pure DSR, 1 = pure net return (default 0).
    exposure_penalty:   Per-step penalty for under-exposure: penalty * (1 - |position|) (default 0).
    downside_only:      Use only downside variance in DSR (Differential Sortino). Default False.
    trend_bonus:        Per-step reward for aligning position with trend (short_ma vs long_ma). Default 0.
    benchmark_relative: If True, DSR uses excess returns over buy-and-hold instead of raw
                        returns. This penalises the agent for under-participation in bull
                        markets and rewards it for avoiding crashes. Default False.
    bull_benchmark:     If True, excess-return comparison is only applied on up-market
                        days. On down days, absolute returns are used, rewarding
                        crash avoidance without penalising imperfect tracking. Default False.
    exposure_floor:     Minimum effective exposure fraction (0-1). The agent's exposure
                        output is remapped from [0,1] to [floor,1]. Set to e.g. 0.5 to
                        keep the agent at least 50% invested. Default 0.
    obs_features:       List of feature names to include in the observation vector.
                        Defaults to DEFAULT_OBS_FEATURES (all 14 features).
                        Pass a subset list to reduce observation dimensionality.
    hard_selection:     If True, the agent picks ONE strategy per step (argmax of
                        first 4 action dims) instead of soft-blending. Default False.
    regime_guardrail:   If True and hard_selection is enabled, BEAR and VOLATILE
                        regimes may only choose Momentum or Defensive. Default False.
    risk_off_exposure_cap:
                        Optional cap on effective exposure in BEAR and VOLATILE
                        regimes, applied after floor remapping. Default None.
    risk_off_strategy_penalty:
                        Optional per-step penalty applied in BEAR and VOLATILE
                        when the resulting position is long and the policy allocates
                        to Momentum or Breakout. Default 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 20,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage_factor: float = 0.0,
        slippage_bps: float = 1.0,
        slippage_vol_coef: float = 0.0,
        borrow_cost_bps_yr: float = 0.0,
        reward_scaling: float = 1.0,
        max_position: float = 1.0,
        normalize_obs: bool = True,
        return_weight: float = 0.0,
        exposure_penalty: float = 0.0,
        downside_only: bool = False,
        trend_bonus: float = 0.0,
        benchmark_relative: bool = False,
        bull_benchmark: bool = False,
        exposure_floor: float = 0.0,
        obs_features: list[str] | None = None,
        hard_selection: bool = False,
        regime_guardrail: bool = False,
        risk_off_exposure_cap: float | None = None,
        risk_off_strategy_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
        drawdown_threshold: float = 0.0,
        macro_features: pd.DataFrame | None = None,
        selective_macro: bool = False,
    ) -> None:
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage_factor = slippage_factor
        self.slippage_bps = float(max(slippage_bps, 0.0))
        self.slippage_vol_coef = float(max(slippage_vol_coef, 0.0))
        self.borrow_cost_bps_yr = float(max(borrow_cost_bps_yr, 0.0))
        self.reward_scaling = reward_scaling
        self.max_position = max_position
        self.normalize_obs = normalize_obs
        self.return_weight = float(np.clip(return_weight, 0.0, 1.0))
        self.exposure_penalty = float(max(exposure_penalty, 0.0))
        self.downside_only = downside_only
        self.trend_bonus = float(max(trend_bonus, 0.0))
        self.benchmark_relative = benchmark_relative
        self.bull_benchmark = bull_benchmark
        self.exposure_floor = float(np.clip(exposure_floor, 0.0, 1.0))
        self.hard_selection = hard_selection
        self.regime_guardrail = regime_guardrail
        self.risk_off_exposure_cap = (
            None
            if risk_off_exposure_cap is None
            else float(np.clip(risk_off_exposure_cap, 0.0, 1.0))
        )
        self.risk_off_strategy_penalty = float(max(risk_off_strategy_penalty, 0.0))
        self.drawdown_penalty = float(max(drawdown_penalty, 0.0))
        self.drawdown_threshold = float(np.clip(drawdown_threshold, 0.0, 1.0))
        self.selective_macro = bool(selective_macro)

        # Pre-compute features once (all 10 base features, plus optional macro
        # features appended after them). Strategies only consume the first 10
        # via positional indexing, so appending is non-breaking.
        self._feature_eng = FeatureEngineer()
        self._all_features_df = self._feature_eng.compute(self.data, macro=macro_features)
        self._features: np.ndarray = self._all_features_df.values  # (T, 10 + n_macro)
        self._n_base_features = len(FEATURE_NAMES)

        # Observation feature subset (only these go into the RL obs vector).
        # Default: every column produced by FeatureEngineer.compute, so adding
        # macro automatically widens the obs.
        all_feature_names = list(self._all_features_df.columns)
        feature_index = {name: i for i, name in enumerate(all_feature_names)}
        if obs_features is not None:
            self._obs_feature_names = list(obs_features)
        else:
            self._obs_feature_names = all_feature_names
        try:
            self._obs_feature_indices = [feature_index[f] for f in self._obs_feature_names]
        except KeyError as exc:
            raise KeyError(
                f"obs_features references unknown column {exc.args[0]!r}; "
                f"available columns: {all_feature_names}"
            ) from exc
        self._n_obs_features = len(self._obs_feature_indices)
        self._obs_features: np.ndarray = self._features[:, self._obs_feature_indices]  # (T, n_obs)

        # Indices within ``_obs_features`` that correspond to macro columns
        # (i.e. columns that appear after the base feature block in the
        # FeatureEngineer output). Used by selective_macro to zero them when
        # the current regime is BULL / SIDEWAYS.
        self._macro_obs_cols: np.ndarray = np.array(
            [i for i, full_idx in enumerate(self._obs_feature_indices)
             if full_idx >= self._n_base_features],
            dtype=np.int64,
        )

        # Pre-compute average volume for slippage model
        vol_series = self.data["Volume"].astype(float)
        self._avg_volume: np.ndarray = (
            vol_series.rolling(20, min_periods=1).mean().values
        )

        # Regime detector
        self._regime_detector = FeatureRegimeDetector()

        # Strategies
        self._strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            TrendFollowingStrategy(),
        ]

        # Spaces
        obs_dim = lookback_window * self._n_obs_features + _N_PORTFOLIO + _N_REGIME
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 4 strategy weights + 1 exposure scalar
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(_N_STRATEGIES + 1,), dtype=np.float32
        )

        # Observation normalizer
        self._obs_normalizer = _RunningNormalizer((obs_dim,)) if normalize_obs else None

        # Episode state (initialised in reset)
        self._step_idx: int = lookback_window
        self._portfolio_value: float = initial_cash
        self._peak_value: float = initial_cash
        self._position: float = 0.0           # current allocation fraction
        self._return_history: list[float] = []
        self._regime_history: list[MarketRegime] = []
        self._action_history: list[np.ndarray] = []

        # Differential Sharpe ratio state (EMA of first two moments)
        self._ema_return: float = 0.0
        self._ema_return_sq: float = 0.0
        self._dsr_eta: float = 0.01  # EMA decay rate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _get_current_regime(self) -> MarketRegime:
        # Causal: only use bars strictly before the current step (i.e., through
        # close[step_idx - 1]). Using ``: step_idx + 1`` would let the regime
        # see today's bar, while the action/reward operates on the close[t-1]
        # → close[t] return -- a 1-bar lookahead leak.
        window = self.data.iloc[max(0, self._step_idx - 20): self._step_idx]
        if len(window) == 0:
            return MarketRegime.SIDEWAYS
        try:
            regimes = self._regime_detector.predict(window)
            return regimes[-1]
        except Exception:
            return MarketRegime.SIDEWAYS

    def _regime_one_hot(self, regime: MarketRegime) -> np.ndarray:
        vec = np.zeros(_N_REGIME, dtype=np.float32)
        vec[regime.value] = 1.0
        return vec

    def _guardrail_allowed_indices(self, regime: MarketRegime) -> np.ndarray:
        if (
            self.regime_guardrail
            and regime in (MarketRegime.BEAR, MarketRegime.VOLATILE)
        ):
            return np.array([0, 3], dtype=np.int64)
        return np.arange(_N_STRATEGIES, dtype=np.int64)

    def _get_observation(self) -> np.ndarray:
        start = self._step_idx - self.lookback_window
        window = self._obs_features[start: self._step_idx]  # (lookback, n_obs)

        regime = self._get_current_regime()

        # Selective macro: hide macro columns from the agent in calm regimes
        # so the policy can't learn spurious bull-fold reactions to them.
        if (
            self.selective_macro
            and self._macro_obs_cols.size > 0
            and regime in (MarketRegime.BULL, MarketRegime.SIDEWAYS)
        ):
            window = window.copy()
            window[:, self._macro_obs_cols] = 0.0

        window_feats = window.ravel()  # (lookback*n_obs)

        portfolio_feats = np.array([
            self._portfolio_value / self.initial_cash - 1.0,   # cash ratio deviation
            self._position,
            (self._portfolio_value - self.initial_cash) / self.initial_cash,  # unrealised PnL
            self._current_drawdown(),
        ], dtype=np.float32)

        regime_feats = self._regime_one_hot(regime)

        obs = np.concatenate([window_feats.astype(np.float32), portfolio_feats, regime_feats])

        if self._obs_normalizer is not None:
            self._obs_normalizer.update(obs)
            obs = self._obs_normalizer.normalize(obs)

        return obs

    def _current_drawdown(self) -> float:
        self._peak_value = max(self._peak_value, self._portfolio_value)
        if self._peak_value == 0:
            return 0.0
        return (self._peak_value - self._portfolio_value) / self._peak_value

    def _rolling_volatility(self) -> float:
        if len(self._return_history) < 2:
            return 0.0
        recent = self._return_history[-20:]
        return float(np.std(recent))

    def _compute_slippage(self, turnover: float) -> float:
        """Slippage model: half-spread base + adverse-selection vol scaler.

        cost (fraction of NAV) = turnover * (slippage_bps + slippage_vol_coef *\n                                            sigma_t / sigma_long_run) / 1e4

        plus an optional legacy quadratic volume-impact term controlled by\n        ``slippage_factor`` (kept for backward compatibility; default 0).\n        """
        if turnover == 0:
            return 0.0

        cost_bps = self.slippage_bps
        if self.slippage_vol_coef > 0 and len(self._return_history) >= 2:
            recent = np.asarray(self._return_history[-20:], dtype=float)
            sigma_t = float(np.std(recent))
            # Long-run vol on training data: rolling 60d std fallback to total std.
            if len(self._return_history) >= 60:
                long_run = float(np.std(self._return_history[-252:]))
            else:
                long_run = sigma_t
            if long_run > 1e-12:
                cost_bps += self.slippage_vol_coef * (sigma_t / long_run)

        cost = turnover * cost_bps / 1e4

        if self.slippage_factor > 0:
            ref_idx = max(0, self._step_idx - 1)
            avg_vol = self._avg_volume[ref_idx]
            ref_close = float(self.data["Close"].iloc[ref_idx])
            avg_dollar_vol = avg_vol * ref_close
            if avg_dollar_vol > 0:
                trade_value = turnover * self._portfolio_value
                cost += self.slippage_factor * (trade_value / avg_dollar_vol) * turnover

        return float(cost)

    def _compute_borrow_cost(self) -> float:
        """Daily borrow cost on negative (short) positions, in fraction of NAV."""
        if self.borrow_cost_bps_yr <= 0 or self._position >= 0:
            return 0.0
        short_size = -self._position  # > 0
        return float(self.borrow_cost_bps_yr / 1e4 / 252.0 * short_size)

    # ------------------------------------------------------------------
    # Normalizer state management
    # ------------------------------------------------------------------

    def set_training_mode(self, training: bool) -> None:
        """Toggle observation-normalizer updates. Set to False during evaluation
        so the policy sees inputs scaled by the same statistics that were
        present at training time (no test-time distribution drift)."""
        if self._obs_normalizer is not None:
            self._obs_normalizer.training = bool(training)

    def get_normalizer_state(self) -> Optional[dict]:
        """Return a serialisable dict of normalizer state (or None if disabled)."""
        return None if self._obs_normalizer is None else self._obs_normalizer.state_dict()

    def load_normalizer_state(self, state: Optional[dict]) -> None:
        """Restore normalizer statistics produced by ``get_normalizer_state``."""
        if state is None or self._obs_normalizer is None:
            return
        self._obs_normalizer.load_state_dict(state)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_idx = self.lookback_window
        self._portfolio_value = self.initial_cash
        self._peak_value = self.initial_cash
        self._position = 0.0
        self._return_history = []
        self._regime_history = []
        self._action_history = []
        self._ema_return = 0.0
        self._ema_return_sq = 0.0

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.asarray(action, dtype=np.float64).ravel()
        current_regime = self._get_current_regime()
        requested_strategy = int(np.argmax(action_arr[:_N_STRATEGIES]))
        allowed_indices = self._guardrail_allowed_indices(current_regime)
        guardrail_active = len(allowed_indices) < _N_STRATEGIES

        # Split action: first 4 = strategy weights, last = exposure scalar
        raw_exposure = float(np.clip(action_arr[_N_STRATEGIES] if len(action_arr) > _N_STRATEGIES else 1.0, 0.0, 1.0))
        # Remap exposure: [0, 1] → [exposure_floor, 1]
        exposure = self.exposure_floor + (1.0 - self.exposure_floor) * raw_exposure
        if self.risk_off_exposure_cap is not None and current_regime in (MarketRegime.BEAR, MarketRegime.VOLATILE):
            exposure = min(exposure, self.risk_off_exposure_cap)

        # Gather strategy signals using the most recent FINALISED feature row.
        # The agent acts at the open of bar step_idx and earns the close[t-1]
        # → close[t] return, so signals must be computed only from data
        # through close[step_idx - 1]. Using self._features[self._step_idx]
        # would let strategies see today's close (lookahead).
        prev_idx = max(0, self._step_idx - 1)
        current_features_arr = self._features[prev_idx]
        # Strategies index by integer position into the *base* 10 features only.
        # If macro features are appended, slice them off here so positional
        # lookups stay aligned with FEATURE_NAMES.
        strategy_features_arr = current_features_arr[: self._n_base_features]
        # Pass the raw feature array; strategies use index-based access by
        # default (see each strategy's __init__ for the feature index map).
        signals = np.array(
            [s.generate_signal(strategy_features_arr).action for s in self._strategies],
            dtype=np.float64,
        )

        if self.hard_selection:
            # Hard selection: pick ONE strategy (argmax), use its signal directly
            weights = np.zeros(_N_STRATEGIES, dtype=np.float64)
            chosen = int(allowed_indices[np.argmax(action_arr[allowed_indices])])
            weights[chosen] = 1.0
            direction = float(np.tanh(signals[chosen] * 5.0))
        else:
            # Soft blend: softmax weights over all strategies. When the
            # regime guardrail is active we apply a hard MASK that zeros
            # logits of forbidden strategies BEFORE the softmax. This is a
            # principled alternative to ``risk_off_strategy_penalty``: the
            # constraint is enforced exactly, not via reward shaping.
            logits = action_arr[:_N_STRATEGIES].astype(np.float64).copy()
            if guardrail_active:
                mask = np.full(_N_STRATEGIES, -np.inf, dtype=np.float64)
                mask[allowed_indices] = 0.0
                logits = logits + mask
            weights = self._softmax(logits)
            blend = float(np.dot(weights, signals))
            chosen = int(np.argmax(weights))
            direction = float(np.tanh(blend * 5.0))

        combined_signal = float(np.clip(direction * exposure * self.max_position, -self.max_position, self.max_position))

        # Compute daily return from close prices
        prev_close = float(self.data["Close"].iloc[self._step_idx - 1])
        curr_close = float(self.data["Close"].iloc[self._step_idx])
        market_return = (curr_close - prev_close) / prev_close

        # Portfolio return = position × market_return
        portfolio_return = self._position * market_return

        # Transaction cost and slippage based on turnover
        turnover = abs(combined_signal - self._position)
        tc_penalty = self.transaction_cost * turnover
        slippage = self._compute_slippage(turnover)
        # Borrow cost is charged on the *prior* short position (before update).
        borrow_cost = self._compute_borrow_cost()

        # Update position
        self._position = combined_signal

        # Net return after costs is the actual change in portfolio value.
        net_return = portfolio_return - tc_penalty - slippage - borrow_cost

        # Update portfolio value first so drawdown reflects current state
        self._portfolio_value *= (1.0 + net_return)
        # Track NET (post-cost) returns. Using gross returns here would inflate
        # downstream Sharpe/Sortino/win-rate metrics computed by the Backtester.
        self._return_history.append(net_return)

        # Differential Sharpe / Sortino ratio reward
        # Updates EMA of return and return^2/ downside^2, then computes the
        # incremental change in the implied ratio.

        # Benchmark-relative mode: use excess return over buy-and-hold
        # so being uninvested during a bull market is penalised.
        if self.bull_benchmark:
            # Asymmetric: only penalise under-participation on up-market days.
            # On down days, reward absolute returns (crash avoidance).
            dsr_input = (net_return - market_return) if market_return > 0 else net_return
        elif self.benchmark_relative:
            dsr_input = net_return - market_return
        else:
            dsr_input = net_return

        eta = self._dsr_eta
        delta_ema = dsr_input - self._ema_return

        if self.downside_only:
            # Sortino variant: only track squared downside returns
            downside_sq = min(dsr_input, 0.0) ** 2
            delta_ema_sq = downside_sq - self._ema_return_sq
        else:
            delta_ema_sq = dsr_input ** 2 - self._ema_return_sq

        # Update EMAs
        self._ema_return += eta * delta_ema
        self._ema_return_sq += eta * delta_ema_sq

        # DSR: d(Sharpe)/dt approximation (or d(Sortino)/dt)
        denom = self._ema_return_sq - self._ema_return ** 2
        if self.downside_only:
            # For Sortino variant, denominator is just EMA of downside^2
            denom = self._ema_return_sq
        if denom > 1e-12:
            dsr = (self._ema_return_sq * delta_ema - 0.5 * self._ema_return * delta_ema_sq) / (denom ** 1.5)
        else:
            dsr = delta_ema  # Fall back to simple return when variance is near zero

        # Blend DSR with raw return
        blended = (1.0 - self.return_weight) * dsr + self.return_weight * net_return

        # Exposure penalty: encourage being invested
        exp_pen = self.exposure_penalty * (1.0 - abs(self._position))

        # Trend bonus: reward being positioned with the prevailing trend
        trend_reward = 0.0
        if self.trend_bonus > 0:
            current_features_dict = dict(zip(FEATURE_NAMES, current_features_arr[: self._n_base_features]))
            sma_short = current_features_dict.get("short_ma", 0.0)
            sma_long = current_features_dict.get("long_ma", 0.0)
            trend_sign = 1.0 if sma_short > sma_long else -1.0
            trend_reward = self.trend_bonus * self._position * trend_sign

        risk_off_penalty = 0.0
        if (
            self.risk_off_strategy_penalty > 0.0
            and current_regime in (MarketRegime.BEAR, MarketRegime.VOLATILE)
            and self._position > 0.0
        ):
            risky_weight = float(np.sum(weights[_RISK_OFF_LONG_STRATEGY_INDICES]))
            risk_off_penalty = self.risk_off_strategy_penalty * risky_weight * self._position

        # Drawdown-aware penalty: apply a CVaR-style cost proportional to the
        # current drawdown beyond ``drawdown_threshold``. Encourages the agent
        # to de-risk after losses instead of doubling down.
        dd_penalty = 0.0
        if self.drawdown_penalty > 0.0:
            current_dd = self._current_drawdown()  # in [0, 1]
            excess_dd = max(current_dd - self.drawdown_threshold, 0.0)
            dd_penalty = self.drawdown_penalty * excess_dd

        reward = float(
            (blended - exp_pen + trend_reward - risk_off_penalty - dd_penalty)
            * self.reward_scaling
        )

        # Track history
        self._regime_history.append(current_regime)
        self._action_history.append(weights.copy())

        self._step_idx += 1
        terminated = self._step_idx >= len(self.data)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "market_return": market_return,
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "transaction_cost": tc_penalty,
            "slippage": slippage,
            "borrow_cost": borrow_cost,
            "regime": current_regime,
            "drawdown": self._current_drawdown(),
            "strategy_weights": weights.copy(),
            "strategy_signals": signals.copy(),
            "requested_strategy": requested_strategy,
            "chosen_strategy": chosen,
            "effective_exposure": exposure,
            "raw_exposure": raw_exposure,
            "guardrail_active": guardrail_active,
            "risk_off_strategy_penalty_applied": risk_off_penalty,
        }
        return obs, reward, terminated, truncated, info
