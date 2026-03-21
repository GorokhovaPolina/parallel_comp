import matplotlib.pyplot as plt
import numpy as np

# Данные из экспериментов (Release)
N = [64, 128, 256, 512, 1024, 2048, 4096]
classic = [2.4502, 1.7116, 1.1625, 0.8848, 0.2096, 0.1581, 0.1519]
transposed = [2.3083, 1.9685, 1.8871, 1.8859, 1.7077, 1.6359, 1.6360]
buffered_m4 = [1.7233, 1.6158, 1.7744, 1.7877, 1.5820, 1.6785, 1.5168]
blocked_s64_m4 = [1.1588, 1.1377, 1.1807, 1.1751, 0.9492, 1.1142, 1.0415]

# Данные для буферизованного (зависимость от M, N=1024)
M_buff = [1, 2, 4, 8, 16]
gflops_buff = [0.6754, 0.7684, 1.4524, 1.6474, 1.6813]

# Данные для блочного (зависимость от S и M, N=1024)
S = [1, 2, 4, 8, 16, 32, 64, 128, 256]
gflops_blocked = {
    'M=1':  [0.02, 0.09, 0.21, 0.30, 0.39, 0.48, 0.49, 0.51, 0.51],
    'M=2':  [0.04, 0.10, 0.29, 0.44, 0.66, 0.68, 0.75, 0.78, 0.79],
    'M=4':  [0.04, 0.13, 0.33, 0.58, 0.87, 1.02, 0.87, 1.14, 1.20],
    'M=8':  [0.04, 0.13, 0.55, 0.54, 1.01, 1.29, 1.06, 1.50, 1.47],
    'M=16': [0.04, 0.13, 0.44, 0.85, 1.09, 1.51, 1.53, 1.64, 1.77]
}

# 1. Сравнение алгоритмов в Release (N)
plt.figure(figsize=(10, 6))
plt.loglog(N, classic, 'o-', label='Classic')
plt.loglog(N, transposed, 's-', label='Transposed B')
plt.loglog(N, buffered_m4, '^-', label='Buffered (M=4)')
plt.loglog(N, blocked_s64_m4, 'd-', label='Blocked (S=64, M=4)')
plt.xlabel('Matrix size N')
plt.ylabel('Performance (GFLOPS)')
plt.title('Performance of matrix multiplication algorithms (Release)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('perf_vs_N_release.png', dpi=150)
plt.show()

# 2. Буферизованный: зависимость от M (N=1024)
plt.figure(figsize=(8, 5))
plt.plot(M_buff, gflops_buff, 'o-', linewidth=2, markersize=8)
plt.xlabel('Unroll factor M')
plt.ylabel('GFLOPS')
plt.title('Buffered multiplication: performance vs unroll factor (N=1024)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(M_buff)
plt.tight_layout()
plt.savefig('buffered_vs_M.png', dpi=150)
plt.show()

# 3. Блочный: зависимость от S для разных M
plt.figure(figsize=(10, 6))
for label, gflops in gflops_blocked.items():
    plt.semilogx(S, gflops, 'o-', label=label)
plt.xlabel('Block size S')
plt.ylabel('GFLOPS')
plt.title('Blocked multiplication: performance vs block size (N=1024)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('blocked_vs_S.png', dpi=150)
plt.show()

# 4. Оптимальная кривая для блочного (лучший M при каждом S)
best_gflops = [max(gflops_blocked[f'M={m}']
                   for m in [1,2,4,8,16]) for i, s in enumerate(S)]
plt.figure(figsize=(8, 5))
plt.semilogx(S, best_gflops, 'o-', color='red', linewidth=2, markersize=8)
plt.xlabel('Block size S')
plt.ylabel('Best GFLOPS (over M)')
plt.title('Blocked multiplication: optimal performance vs block size')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('blocked_optimal_vs_S.png', dpi=150)
plt.show()

print("Графики сохранены: perf_vs_N_release.png, buffered_vs_M.png, blocked_vs_S.png, blocked_optimal_vs_S.png")