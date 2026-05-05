import json
import statistics as s

base = json.load(open('results/wf_weekly_soft.json'))['folds']
full = json.load(open('results/wf_weekly_soft_macro.json'))['folds']
sel  = json.load(open('results/wf_weekly_soft_macro_selective.json'))['folds']

print(f"{'fold':>4} | {'period':<22} | {'base_Shp':>8} {'full_Shp':>8} {'sel_Shp':>8} | {'sel-base':>8} {'sel-full':>8}")
print('-' * 92)
for fb, ff, fs in zip(base, full, sel):
    period = f"{fb['test_start'][:10]}..{fb['test_end'][:10]}"
    print(f"{fb['fold']:>4} | {period} | "
          f"{fb['sharpe_ratio']:+8.2f} {ff['sharpe_ratio']:+8.2f} {fs['sharpe_ratio']:+8.2f} | "
          f"{fs['sharpe_ratio']-fb['sharpe_ratio']:+8.2f} {fs['sharpe_ratio']-ff['sharpe_ratio']:+8.2f}")

print('-' * 92)
def stats(folds, key='sharpe_ratio'):
    vals = [f[key] for f in folds]
    return s.mean(vals), s.median(vals), s.pstdev(vals)

for label, F in [('base', base), ('full', full), ('selective', sel)]:
    m, med, sd = stats(F)
    rets = [f['total_return'] for f in F]
    bnh  = [f['benchmark_sharpe'] for f in F]
    wins = sum(1 for f in F if f['sharpe_ratio'] > f['benchmark_sharpe'])
    print(f"{label:>10}: mean Sharpe={m:+.3f}  median={med:+.3f}  std={sd:.2f}  "
          f"mean ret={sum(rets)/len(rets)*100:+.2f}%  beats B&H={wins}/{len(F)}")

print('\nDelta sel vs full per fold (positive = selective improved):')
ds = [fs['sharpe_ratio']-ff['sharpe_ratio'] for ff, fs in zip(full, sel)]
print(f"  mean={s.mean(ds):+.3f}  median={s.median(ds):+.3f}  wins={sum(1 for d in ds if d>0)}/{len(ds)}")

print('\nDelta sel vs base per fold:')
db = [fs['sharpe_ratio']-fb['sharpe_ratio'] for fb, fs in zip(base, sel)]
print(f"  mean={s.mean(db):+.3f}  median={s.median(db):+.3f}  wins={sum(1 for d in db if d>0)}/{len(db)}")
