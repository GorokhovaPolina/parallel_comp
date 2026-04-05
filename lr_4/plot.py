import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', comment='#', header=None,
                 names=['size_bytes', 'sequential_ns', 'random_ns', 'random_index_ns'])

df['size_kb'] = df['size_bytes'] / 1024

plt.figure(figsize=(12, 6))
plt.semilogx(df['size_kb'], df['sequential_ns'], 'o-', label='Sequential')
plt.semilogx(df['size_kb'], df['random_ns'], 's-', label='Random (on-fly)')
plt.semilogx(df['size_kb'], df['random_index_ns'], '^-', label='Random with index array')
plt.xlabel('Размер данных (КБ)')
plt.ylabel('Время одной итерации (нс)')
plt.title('Латентность подсистемы памяти')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('latency_plot.png', dpi=150)
plt.show()