import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
from statsmodels.tsa.stattools import adfuller
from pandas_datareader import data as web

cre = pd.read_excel('CRE.xlsx')
card = pd.read_excel('card.xlsx')
cre.set_index('date', inplace=True)
card.set_index('date', inplace=True)
cre['pct'] = 100 * cre['chargeoffs'] / cre['loans']
card['pct'] = 100 * card['chargeoffs'] / card['loans']
cre['pct_diff'] = cre['pct'].diff()
card['pct_diff'] = card['pct'].diff()
cre.dropna(inplace=True)
card.dropna(inplace=True)

series = ["UNRATE", "DCOILBRENTEU", "GDP", "T10Y2Y", "VIXCLS"]
data = {s: web.DataReader(s, "fred", start='2000-01-01') for s in series}
unrate, oil, gdp, t10y2y, volatity = [data[s] for s in series]

unrate.index = unrate.index - pd.DateOffset(days=1)
gdp.index = gdp.index - pd.DateOffset(days=1)
unrate = unrate.resample('QE').last()
gdp = gdp.resample('QE').last()
oil = oil.resample('QE').mean()
t10y2y = t10y2y.resample('QE').mean()
volatity = volatity.resample('QE').mean()
econ = pd.concat([unrate, oil, gdp, t10y2y, volatity], axis=1)
econ['gdp_growth'] = econ['GDP'].pct_change(fill_method=None)
econ.dropna(inplace=True)

for col in ['UNRATE', 'DCOILBRENTEU', 'GDP', 'gdp_growth', 'T10Y2Y', 'VIXCLS']:
    if adfuller(econ[col])[1] > 0.05 and col != 'GDP':
        econ[col + '_diff'] = econ[col].diff()
    econ[col] = econ[col].shift(-1)

cre.columns = ['cre_' + col for col in cre.columns]
card.columns = ['card_' + col for col in card.columns]
df = pd.merge(cre, card, left_index=True, right_index=True)
df = pd.merge(df, econ, left_index=True, right_index=True)
df = df[['cre_loans', 'cre_chargeoffs', 'cre_pct', 'cre_pct_diff',
         'card_loans', 'card_chargeoffs', 'card_pct', 'card_pct_diff',
         'UNRATE', 'DCOILBRENTEU', 'DCOILBRENTEU_diff', 'GDP', 'gdp_growth', 'T10Y2Y', 'VIXCLS']]
dfstat = df[['cre_pct_diff', 'card_pct_diff', 'UNRATE', 'DCOILBRENTEU_diff', 'gdp_growth', 'T10Y2Y', 'VIXCLS']]

def lag_and_regression(dependent_var, explanatory_vars, df):
    df.loc[:, dependent_var + '_lag1'] = df[dependent_var].shift(1)
    dfc = df.dropna()
    X = dfc[[dependent_var + '_lag1'] + list(explanatory_vars)].dropna()
    y = dfc[dependent_var]
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit().rsquared

explanatory_vars = ['UNRATE', 'DCOILBRENTEU_diff', 'gdp_growth', 'T10Y2Y', 'VIXCLS']
factor_combinations = list(itertools.combinations(explanatory_vars, 3))
results = [{'combo': combo,
            'r2_cre': lag_and_regression('cre_pct_diff', combo, dfstat),
            'r2_card': lag_and_regression('card_pct_diff', combo, dfstat)}
           for combo in factor_combinations]

best_cre = sorted(results, key=lambda x: x['r2_cre'], reverse=True)[0]
best_card = sorted(results, key=lambda x: x['r2_card'], reverse=True)[0]
print(best_cre['combo'], best_cre['r2_cre'])
print(best_card['combo'], best_card['r2_card'])
