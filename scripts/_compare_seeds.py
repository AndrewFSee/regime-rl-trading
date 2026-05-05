"""Aggregate weekly+macro walk-forward across seeds 0..3."""
import json
import statistics as s

paths = {
    0: 'results/wf_weekly_soft_macro.json',
    1: 'results/wf_weekly_macro_seed1.json',
    2: 'results/wf_weekly_macro_seed2.json',
    3: 'results/wf_weekly_macro_seed3.json',
}
data = {seed: json.load(open(p))['folds'] for seed, p in paths.items()}
n_folds = len(data[0])
seeds = sorted(data.keys())

print(f"{'fold':>4} | {'period':<22} |", end='')
for sd in seeds:
    print(f" s{sd}_Shp ", end='')
print(f"|  {'mean':>5} {'std':>5} {'min':>6} {'max':>6}  | beats_BnH")
print('-' * 110)

# per-fold agg
mean_shp_per_fold = []
mean_ret_per_fold = []
all_per_seed_means = {sd: [] for sd in seeds}
beats_count_per_seed = {sd: 0 for sd in seeds}
for i in range(n_folds):
    period = f"{data[0][i]['test_start'][:10]}..{data[0][i]['test_end'][:10]}"
    shps = [data[sd][i]['sharpe_ratio'] for sd in seeds]
    rets = [data[sd][i]['total_return'] for sd in seeds]
    bnh_shp = data[0][i]['benchmark_sharpe']
    beats = sum(1 for v in shps if v > bnh_shp)
    for sd, v in zip(seeds, shps):
        all_per_seed_means[sd].append(v)
        if v > bnh_shp:
            beats_count_per_seed[sd] += 1
    mean_shp = sum(shps) / len(shps)
    mean_ret = sum(rets) / len(rets)
    mean_shp_per_fold.append(mean_shp)
    mean_ret_per_fold.append(mean_ret)
    print(f"{i:>4} | {period} |", end='')
    for v in shps:
        print(f" {v:+6.2f}", end='')
    print(f" | {mean_shp:+6.2f} {s.pstdev(shps):5.2f} {min(shps):+6.2f} {max(shps):+6.2f} | {beats}/4 vs B&H={bnh_shp:+.2f}")

print('-' * 110)
print(f"\n=== Per-seed aggregate ===")
for sd in seeds:
    vals = all_per_seed_means[sd]
    print(f"  seed {sd}: mean={s.mean(vals):+.3f}  median={s.median(vals):+.3f}  std={s.pstdev(vals):.2f}  beats_B&H={beats_count_per_seed[sd]}/{n_folds}")

print(f"\n=== Cross-seed aggregate (per-fold mean Sharpe) ===")
print(f"  mean of per-fold means : {s.mean(mean_shp_per_fold):+.3f}")
print(f"  median of per-fold means: {s.median(mean_shp_per_fold):+.3f}")
print(f"  std of per-fold means  : {s.pstdev(mean_shp_per_fold):.2f}")
print(f"  mean of per-fold returns: {s.mean(mean_ret_per_fold)*100:+.2f}%")

# folds where ALL seeds beat B&H, and folds where 0/4 do
print(f"\n=== Robust fold analysis ===")
robust_winners = []
robust_losers = []
for i in range(n_folds):
    bnh_shp = data[0][i]['benchmark_sharpe']
    shps = [data[sd][i]['sharpe_ratio'] for sd in seeds]
    if all(v > bnh_shp for v in shps):
        robust_winners.append((i, data[0][i]['test_start'][:10], shps, bnh_shp))
    if all(v <= bnh_shp for v in shps) and bnh_shp > 0:
        robust_losers.append((i, data[0][i]['test_start'][:10], shps, bnh_shp))

print(f"  Folds where ALL 4 seeds beat B&H: {len(robust_winners)}")
for i, dt, shps, bnh in robust_winners:
    print(f"    fold {i:>2} ({dt}): shps={[f'{v:+.2f}' for v in shps]}  B&H={bnh:+.2f}")

# Crisis vs bull breakdown
crisis_folds = [0, 5, 6, 7, 8, 12, 17, 19]  # recovery / crisis / bear
bull_folds = [1, 2, 3, 4, 10, 11, 13, 14, 15, 16, 18, 20]
def avg_per_fold_mean(idx_list):
    return s.mean([mean_shp_per_fold[i] for i in idx_list])
print(f"\n=== Crisis/recovery folds {crisis_folds}: avg mean-Sharpe = {avg_per_fold_mean(crisis_folds):+.3f}")
print(f"=== Bull folds          {bull_folds}: avg mean-Sharpe = {avg_per_fold_mean(bull_folds):+.3f}")
