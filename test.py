import statsmodels.api as sm

# Example: 40 successes out of 100 trials
successes, trials = 40, 100
confidence_level = 0.95

# Clopper-Pearson Interval
cp_interval = sm.stats.proportion_confint(successes, trials, alpha=1-confidence_level, method='binom_test')

# Wilson Score Interval
wilson_interval = sm.stats.proportion_confint(successes, trials, alpha=1-confidence_level, method='wilson')

print(f"Clopper-Pearson Interval: {cp_interval}")
print(f"Wilson Score Interval: {wilson_interval}")
