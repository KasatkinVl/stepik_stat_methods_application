from scipy.stats import chi2_contingency

observed = ([20,15],[11,12],[7,9])
chi2, p, df, expected = chi2_contingency(observed)
print ('x-squared =', chi2)
print ('p-value =', p)
print ('df =', df)
print('expected =', expected)

# print(chi2_contingency(observed))
