import re

p = '<your_log>.txt'

with open(p, 'r') as f:
    s = ''.join(f.readlines())
    acc_results = re.findall(r'Top-1 Accuracy: (\d+\.\d+)', s)
    acc_results = [float(_) for _ in acc_results]

acc_final_results, ece_final_results = [], []
for i in range(10):
    acc_final_results.append(sum(acc_results[i*15:(i+1)*15])/15)
    print(acc_results[i*15:(i+1)*15])


print(acc_final_results)
# print(acc_results)
