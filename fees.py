import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d = df.copy()
n, tot, ex, fee = ('number_of_claims', 'total_claim_cost',
                   'total_claim_cost_excluding_fees', 'total_fees')

# 1. Data checks
print('Rows where total != ex-fees + fees:',
      (~np.isclose(d[tot], d[ex] + d[fee], atol=0.01)).sum())
print('Rows with number_of_claims != 1:', (d[n] != 1).sum())
print('Rows with negative ex-fees cost:', (d[ex] < 0).sum())

# 2. Keep only real claims: positive indemnity (drops fee-only and net-negative rows)
d = d[(d[ex] > 0) & (d[n] > 0)]

# 3. Band on per-claim indemnity (excluding fees)
bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
        1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000,
        8000, 9000, 10000, 15000, 20000, 25000, 50000, 75000, 100000, 200000,
        300000, 500000, 1000000, np.inf]
d['band'] = pd.cut(d[ex] / d[n], bins=bins)
d['row_fee_pct'] = d[fee] / d[ex]

# 4. Aggregate as ratios of sums
g = d.groupby('band', observed=True).agg(
    n_claims=(n, 'sum'),
    fees=(fee, 'sum'),
    indemnity=(ex, 'sum'),
    median_row_pct=('row_fee_pct', 'median'),
)
g['fee_pct'] = g['fees'] / g['indemnity']   # fees as a share of indemnity
g['avg_fee'] = g['fees'] / g['n_claims']    # average fee per claim, £

# 5. Plot
labels = g.index.astype(str)
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_bar(x=labels, y=g['n_claims'], name='Claim count', opacity=0.5,
            secondary_y=True)
fig.add_scatter(x=labels, y=g['fee_pct'], name='Fee % (ratio of sums)',
                mode='lines+markers')
fig.add_scatter(x=labels, y=g['median_row_pct'], name='Fee % (median row)',
                mode='lines', line=dict(dash='dot'))
fig.update_yaxes(tickformat='.0%', secondary_y=False, title='Fee %')
fig.update_yaxes(title='Claim count', secondary_y=True)
fig.show()

# 6. Fixed + % check: average fee £ by band
px_fee = go.Figure(go.Scatter(x=labels, y=g['avg_fee'], mode='lines+markers'))
px_fee.update_layout(yaxis_title='Average fee per claim (£)')
px_fee.show()


# Add this to the aggregation:
d['has_fee'] = d[fee] > 0

g = d.groupby('band', observed=True).agg(
    n_claims=(n, 'sum'),
    n_with_fee=('has_fee', 'sum'),
    fees=(fee, 'sum'),
    indemnity=(ex, 'sum'),
)
g['pct_with_fee'] = g['n_with_fee'] / g['n_claims']
g['fee_pct'] = g['fees'] / g['indemnity']                   # cost basis, for loadings
g['avg_fee_when_charged'] = g['fees'] / g['n_with_fee']     # severity of fee

# Median among claims that do have a fee (median across all rows will often be 0)
g['median_pct_when_charged'] = (d[d['has_fee']]
                                .groupby('band', observed=True)['row_fee_pct']
                                .median())

# Recording consistency check
d.groupby(['coverholder', 'year'])['has_fee'].mean().unstack()