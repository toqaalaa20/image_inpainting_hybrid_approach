#!/usr/bin/env python3
"""
Generate a non-interactive summary report (PNG) showing mean PSNR/LPIPS per method and a bar chart of counts.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

metrics_csv = Path('results/final_metrics.csv')
out_png = Path('results/summary_report.png')

if not metrics_csv.exists():
    print('Metrics CSV not found:', metrics_csv)
    raise SystemExit(1)

df = pd.read_csv(metrics_csv)
# convert columns to numeric where possible
for col in ['psnr', 'lpips']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# group by method
group = df.groupby('method').agg({'psnr': 'mean', 'lpips': 'mean', 'image': 'count'}).rename(columns={'image': 'count'})
print(group)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
group['psnr'].sort_values(ascending=False).plot(kind='bar', ax=axes[0], title='Mean PSNR')
group['lpips'].sort_values(ascending=True).plot(kind='bar', ax=axes[1], title='Mean LPIPS')
group['count'].sort_values(ascending=False).plot(kind='bar', ax=axes[2], title='Samples per method')
plt.tight_layout()
fig.savefig(out_png)
print('Saved summary report to', out_png)
