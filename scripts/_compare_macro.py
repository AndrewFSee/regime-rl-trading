import json
b = json.load(open('results/wf_weekly_soft.json'))['folds']
m = json.load(open('results/wf_weekly_soft_macro.json'))['folds']
print('fold | period                  | base_ret  base_Shp |  macro_ret macro_Shp | dShp')
print('-' * 92)
for fb, fm in zip(b, m):
    period = f"{fb['test_start'][:10]}..{fb['test_end'][:10]}"
    print(f"{fb['fold']:>4} | {period} | {fb['total_return']*100:+7.2f}% {fb['sharpe_ratio']:+6.2f}    | {fm['total_return']*100:+7.2f}% {fm['sharpe_ratio']:+6.2f}     | {fm['sharpe_ratio']-fb['sharpe_ratio']:+5.2f}")

import statistics as s
db = [fm['sharpe_ratio'] - fb['sharpe_ratio'] for fb, fm in zip(b, m)]
print('-' * 92)
print(f"mean dShp = {s.mean(db):+.3f}   median dShp = {s.median(db):+.3f}   wins (macro>base) = {sum(1 for d in db if d>0)}/{len(db)}")
