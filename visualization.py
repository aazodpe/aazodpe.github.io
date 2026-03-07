"""
Visualization Script - Grid Emissions & Demand Analysis
Group 4 - Milestone 2

Generates 10 exploratory visualizations from cleaned data and saves as PNG.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG   = '#0a0e1a'
CARD_BG   = '#1a1f2e'
CYAN      = '#00d9ff'
ORANGE    = '#ff6b35'
YELLOW    = '#ffd23f'
TEXT      = '#e8edf4'
MUTED     = '#9ca9c0'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   MUTED,
    'axes.labelcolor':  TEXT,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT,
    'grid.color':       '#2a3040',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  TEXT,
    'figure.dpi':       150,
})

SAVE_DIR = 'figures'

def save(fig, name):
    path = f'{SAVE_DIR}/{name}'
    fig.savefig(path, bbox_inches='tight', facecolor=DARK_BG, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


# ── Load data ────────────────────────────────────────────────────────────────
e = pd.read_csv('data/processed/emissions_clean.csv', parse_dates=['timestamp'])
d = pd.read_csv('data/processed/demand_clean.csv',   parse_dates=['timestamp'])

# Demand was hourly before 15-min resampling → 3/4 rows are NaN; strip them for plotting
d_hourly = d.dropna(subset=['demand_MW']).copy()

# Event window (heat wave peak: Aug 15-18)
event_start = pd.Timestamp('2024-08-15', tz='UTC')
event_end   = pd.Timestamp('2024-08-19', tz='UTC')
e_event = e[(e['timestamp'] >= event_start) & (e['timestamp'] < event_end)]
d_event = d_hourly[(d_hourly['timestamp'] >= event_start) & (d_hourly['timestamp'] < event_end)]

# Merged dataset for viz 03 & 10: join on nearest timestamp, drop NaN only for key columns
merged = pd.merge_asof(
    e[['timestamp','value','hour','day_of_week','is_weekend',
       'value_rolling_mean_24h','value_rolling_std_24h']].sort_values('timestamp'),
    d_hourly[['timestamp','demand_MW']].sort_values('timestamp'),
    on='timestamp', direction='nearest', tolerance=pd.Timedelta('30min')
).dropna(subset=['value', 'demand_MW'])


# ── VIZ 01: Time Series – Demand & Emissions During Heat Wave ────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle('Viz 01 · Time Series: Emissions & Demand During Aug 2024 Heat Wave',
             color=TEXT, fontsize=13, fontweight='bold', y=1.01)

# Emissions (15-min resolution, no NaN)
ax1.plot(e['timestamp'], e['value'], color=CYAN, lw=1.2, alpha=0.7, label='CO₂ MOER')
ax1.axvspan(event_start, event_end, color=ORANGE, alpha=0.15, label='Heat Wave')
ax1.set_ylabel('CO₂ MOER (lbs/MWh)', color=TEXT)
ax1.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
ax1.grid(True)

# Demand: plot only valid hourly readings (no NaN gaps)
ax2.plot(d_hourly['timestamp'], d_hourly['demand_MW'], color=YELLOW, lw=1.2, alpha=0.7, label='Demand (MWh)')
ax2.axvspan(event_start, event_end, color=ORANGE, alpha=0.15, label='Heat Wave')
ax2.set_ylabel('Demand (MWh)', color=TEXT)
ax2.set_xlabel('Date (UTC)', color=TEXT)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=30)
ax2.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
ax2.grid(True)

plt.tight_layout()
save(fig, 'viz01_timeseries.png')


# ── VIZ 02: Distribution – Emissions Intensity ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Viz 02 · Distribution Analysis: CO₂ Emissions Intensity (CAISO North)',
             color=TEXT, fontsize=13, fontweight='bold')

# Histogram
axes[0].hist(e['value'].dropna(), bins=40, color=CYAN, alpha=0.8, edgecolor=DARK_BG)
axes[0].axvline(e['value'].mean(), color=ORANGE, lw=2, linestyle='--', label=f'Mean: {e["value"].mean():.0f}')
axes[0].axvline(e['value'].median(), color=YELLOW, lw=2, linestyle=':', label=f'Median: {e["value"].median():.0f}')
axes[0].set_xlabel('CO₂ MOER (lbs/MWh)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Emissions Distribution')
axes[0].legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
axes[0].grid(True)

# Box plot split by period (event vs baseline)
e_copy = e.copy()
e_copy['period'] = np.where(
    (e_copy['timestamp'] >= event_start) & (e_copy['timestamp'] < event_end),
    'Heat Wave', 'Baseline'
)
groups = [e_copy[e_copy['period'] == p]['value'].dropna() for p in ['Baseline', 'Heat Wave']]
bp = axes[1].boxplot(groups, labels=['Baseline', 'Heat Wave'], patch_artist=True,
                     medianprops=dict(color=DARK_BG, lw=2))
colors = [CYAN, ORANGE]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_ylabel('CO₂ MOER (lbs/MWh)')
axes[1].set_title('Baseline vs Heat Wave')
axes[1].grid(True, axis='y')

plt.tight_layout()
save(fig, 'viz02_distribution.png')


# ── VIZ 03: Correlation Heatmap ──────────────────────────────────────────────
# Use the pre-built merged dataframe (demand_MW_rolling_mean_24h is all-NaN so excluded)
corr_cols = ['value','demand_MW','hour','day_of_week','is_weekend',
             'value_rolling_mean_24h','value_rolling_std_24h']
corr_labels = ['CO₂ MOER','Demand MW','Hour','Day of Week','Is Weekend',
               'MOER Roll Mean 24h','MOER Roll Std 24h']
corr = merged[corr_cols].dropna().corr()
corr.index = corr_labels
corr.columns = corr_labels

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Viz 03 · Correlation Heatmap: Feature Relationships',
             color=TEXT, fontsize=13, fontweight='bold')
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9, 'color': TEXT})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
save(fig, 'viz03_correlation_heatmap.png')


# ── VIZ 04: Seasonal Decomposition ──────────────────────────────────────────
# Use hourly demand (resample from 15-min)
d_hourly = d.set_index('timestamp')['demand_MW'].resample('1h').mean().dropna()
decomp = seasonal_decompose(d_hourly, model='additive', period=24)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Viz 04 · Seasonal Decomposition: Demand Patterns (Hourly)',
             color=TEXT, fontsize=13, fontweight='bold')

components = [
    (d_hourly,         'Observed',  CYAN),
    (decomp.trend,     'Trend',     YELLOW),
    (decomp.seasonal,  'Seasonal',  ORANGE),
    (decomp.resid,     'Residual',  MUTED),
]
for ax, (data, label, color) in zip(axes, components):
    ax.plot(data.index, data.values, color=color, lw=1.2)
    ax.axvspan(event_start, event_end, color=ORANGE, alpha=0.15)
    ax.set_ylabel(label, color=TEXT, fontsize=10)
    ax.grid(True)

axes[-1].set_xlabel('Date (UTC)', color=TEXT)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=30)
plt.tight_layout()
save(fig, 'viz04_seasonal_decomp.png')


# ── VIZ 05: Scatter – Daily Peak Demand vs Peak Emissions ────────────────────
e_daily = e.copy()
e_daily['date'] = e_daily['timestamp'].dt.date
d_daily = d.copy()
d_daily['date'] = d_daily['timestamp'].dt.date

daily = pd.merge(
    e_daily.groupby('date')['value'].max().rename('peak_moer'),
    d_daily.groupby('date')['demand_MW'].max().rename('peak_demand'),
    on='date'
).reset_index()
daily['is_event'] = daily['date'].astype(str).between('2024-08-15', '2024-08-18')

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Viz 05 · Scatter: Daily Peak Demand vs. Peak CO₂ Emissions',
             color=TEXT, fontsize=13, fontweight='bold')

baseline = daily[~daily['is_event']]
event    = daily[daily['is_event']]

ax.scatter(baseline['peak_demand'], baseline['peak_moer'],
           color=CYAN, alpha=0.75, s=70, label='Baseline Days', zorder=3)
ax.scatter(event['peak_demand'],    event['peak_moer'],
           color=ORANGE, alpha=0.9, s=100, marker='*', label='Heat Wave Days', zorder=4)

# Trend line
m, b, r, *_ = stats.linregress(daily['peak_demand'], daily['peak_moer'])
x_line = np.linspace(daily['peak_demand'].min(), daily['peak_demand'].max(), 100)
ax.plot(x_line, m * x_line + b, color=YELLOW, lw=1.5, linestyle='--',
        label=f'Trend (r={r:.2f})')

ax.set_xlabel('Peak Demand (MWh)', color=TEXT)
ax.set_ylabel('Peak CO₂ MOER (lbs/MWh)', color=TEXT)
ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
ax.grid(True)
plt.tight_layout()
save(fig, 'viz05_scatter_peak.png')


# ── VIZ 06: Histogram – Emissions Distribution Detail ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Viz 06 · Distribution Detail: Emissions Before, During & After Heat Wave',
             color=TEXT, fontsize=13, fontweight='bold')

before = e[e['timestamp'] < event_start]['value'].dropna()
during = e[(e['timestamp'] >= event_start) & (e['timestamp'] < event_end)]['value'].dropna()
after  = e[e['timestamp'] >= event_end]['value'].dropna()

axes[0].hist(before, bins=30, alpha=0.6, color=CYAN,   label=f'Before (n={len(before)})', density=True)
axes[0].hist(during, bins=30, alpha=0.6, color=ORANGE, label=f'During (n={len(during)})', density=True)
axes[0].hist(after,  bins=30, alpha=0.6, color=YELLOW, label=f'After  (n={len(after)})',  density=True)
axes[0].set_xlabel('CO₂ MOER (lbs/MWh)')
axes[0].set_ylabel('Density')
axes[0].set_title('Overlapping Histograms')
axes[0].legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
axes[0].grid(True)

# Daily range (max - min) as proxy for volatility
e_daily2 = e.copy()
e_daily2['date'] = e_daily2['timestamp'].dt.date
daily_range = e_daily2.groupby('date')['value'].agg(lambda x: x.max() - x.min()).reset_index()
daily_range.columns = ['date', 'daily_range']
is_event = daily_range['date'].astype(str).between('2024-08-15', '2024-08-18')
axes[1].bar(range(len(daily_range)), daily_range['daily_range'],
            color=[ORANGE if v else CYAN for v in is_event], alpha=0.8)
axes[1].set_xlabel('Day Index')
axes[1].set_ylabel('Daily CO₂ Range (lbs/MWh)')
axes[1].set_title('Daily Emissions Volatility')
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=CYAN, label='Baseline'), Patch(color=ORANGE, label='Heat Wave')],
               facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
axes[1].grid(True, axis='y')
plt.tight_layout()
save(fig, 'viz06_histogram_duration.png')


# ── VIZ 07: Q-Q Plot – Emissions Normality ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Viz 07 · Q-Q Plot: Emissions Normality Assessment',
             color=TEXT, fontsize=13, fontweight='bold')

for ax, (data, label, color) in zip(axes, [
    (e['value'].dropna(), 'All Periods', CYAN),
    (during,              'Heat Wave',   ORANGE),
]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data)
    ax.scatter(osm, osr, color=color, alpha=0.5, s=15, label=f'n={len(data)}')
    fit_x = np.array([min(osm), max(osm)])
    ax.plot(fit_x, slope * fit_x + intercept, color=YELLOW, lw=2, linestyle='--',
            label=f'Normal fit (r={r:.3f})')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(label)
    ax.legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
    ax.grid(True)

plt.tight_layout()
save(fig, 'viz07_qq_plot.png')


# ── VIZ 08: Heatmap – Hourly Emissions by Day of Week ───────────────────────
pivot = e.groupby(['day_of_week', 'hour'])['value'].mean().unstack()
pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Viz 08 · Heatmap: Average CO₂ MOER by Hour & Day of Week',
             color=TEXT, fontsize=13, fontweight='bold')

cmap2 = sns.color_palette('YlOrRd', as_cmap=True)
sns.heatmap(pivot, cmap=cmap2, ax=ax, annot=True, fmt='.0f',
            linewidths=0.3, cbar_kws={'label': 'CO₂ MOER (lbs/MWh)', 'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_xlabel('Hour of Day (UTC)', color=TEXT)
ax.set_ylabel('Day of Week', color=TEXT)
plt.tight_layout()
save(fig, 'viz08_hourly_heatmap.png')


# ── VIZ 09: Cumulative Emissions Over Event Period ───────────────────────────
e_cum = e.copy().sort_values('timestamp')
# Convert lbs/MWh to approximate cumulative (lbs, assuming ~30k MW avg grid)
e_cum['cum_moer'] = e_cum['value'].cumsum()
e_cum_event = e_cum[(e_cum['timestamp'] >= event_start) & (e_cum['timestamp'] < event_end)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Viz 09 · Cumulative Emissions Impact',
             color=TEXT, fontsize=13, fontweight='bold')

axes[0].fill_between(e_cum['timestamp'], e_cum['cum_moer'],
                     color=CYAN, alpha=0.4)
axes[0].plot(e_cum['timestamp'], e_cum['cum_moer'], color=CYAN, lw=1.5)
axes[0].axvspan(event_start, event_end, color=ORANGE, alpha=0.2, label='Heat Wave')
axes[0].set_ylabel('Cumulative CO₂ MOER (lbs/MWh sum)')
axes[0].set_xlabel('Date (UTC)')
axes[0].set_title('Full Period')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)
axes[0].legend(facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
axes[0].grid(True)

# Daily total emissions (sum of MOER readings)
e_daily3 = e.copy()
e_daily3['date'] = e_daily3['timestamp'].dt.date
daily_sum = e_daily3.groupby('date')['value'].sum().reset_index()
is_event3 = daily_sum['date'].astype(str).between('2024-08-15','2024-08-18')
bars = axes[1].bar(range(len(daily_sum)), daily_sum['value'],
                   color=[ORANGE if v else CYAN for v in is_event3], alpha=0.85)
axes[1].set_xlabel('Day Index')
axes[1].set_ylabel('Daily Total CO₂ (lbs/MWh sum)')
axes[1].set_title('Daily Emissions Total')
axes[1].legend(handles=[Patch(color=CYAN, label='Baseline'), Patch(color=ORANGE, label='Heat Wave')],
               facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
axes[1].grid(True, axis='y')

plt.tight_layout()
save(fig, 'viz09_cumulative.png')


# ── VIZ 10: Pair Plot – Multivariate Relationships ──────────────────────────
pair_data = merged[['timestamp','value','demand_MW','hour','day_of_week','value_rolling_mean_24h']].dropna().copy()
pair_data['Period'] = np.where(
    (pair_data['timestamp'] >= event_start) & (pair_data['timestamp'] < event_end),
    'Heat Wave', 'Baseline'
)
pair_data = pair_data.drop(columns='timestamp')
pair_data.columns = ['CO₂ MOER','Demand MW','Hour','Day','MOER 24h Avg','Period']

palette = {'Baseline': CYAN, 'Heat Wave': ORANGE}
g = sns.pairplot(pair_data, hue='Period', palette=palette,
                 plot_kws={'alpha': 0.4, 's': 15},
                 diag_kind='kde', corner=True)
g.figure.suptitle('Viz 10 · Pair Plot: Multivariate Feature Relationships',
                  color=TEXT, fontsize=13, fontweight='bold', y=1.02)
g.figure.set_facecolor(DARK_BG)
for ax in g.axes.flatten():
    if ax:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color='#2a3040', linestyle='--', alpha=0.5)

g.figure.savefig(f'{SAVE_DIR}/viz10_pairplot.png',
                 bbox_inches='tight', facecolor=DARK_BG, dpi=150)
plt.close()
print(f'Saved {SAVE_DIR}/viz10_pairplot.png')

print('\nAll 10 visualizations generated successfully.')
