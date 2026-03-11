import csv
from collections import defaultdict

data = []
with open('customer_churn_dataset-testing-master.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row or not row[0]: continue
        data.append(row)
        
payment_delay_idx = header.index('Payment Delay')
churn_idx = header.index('Churn')

delay_churn = {}
for row in data:
    try:
        delay = int(row[payment_delay_idx])
        churn = int(row[churn_idx])
        
        if delay not in delay_churn:
            delay_churn[delay] = {'total': 0, 'churn': 0}
        
        delay_churn[delay]['total'] += 1
        delay_churn[delay]['churn'] += churn
    except ValueError:
        pass

import json
output = {'payment_delay': {}, 'perfect_predictors': {}}

for delay in sorted(delay_churn.keys()):
    stats = delay_churn[delay]
    churn_rate = stats['churn'] / stats['total'] * 100
    output['payment_delay'][delay] = {'churn_rate': churn_rate, 'total': stats['total']}
    
for col in ['Usage Frequency', 'Support Calls', 'Last Interaction', 'Tenure']:
    col_idx = header.index(col)
    stats = defaultdict(lambda: {'total': 0, 'churn': 0})
    
    for row in data:
        try:
            val = float(row[col_idx])
            churn = int(row[churn_idx])
            stats[val]['total'] += 1
            stats[val]['churn'] += churn
        except ValueError:
            pass
            
    perfect_flags = []
    for val in sorted(stats.keys()):
        if stats[val]['total'] > 10: # Only count statistically significant thresholds
            ch_rate = stats[val]['churn'] / stats[val]['total'] * 100
            if ch_rate > 95 or ch_rate < 5:
                perfect_flags.append({'val': val, 'churn_rate': ch_rate, 'total': stats[val]['total']})
    
    output['perfect_predictors'][col] = perfect_flags

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=4)
