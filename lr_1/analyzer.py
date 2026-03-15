import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./data/data_optimized.txt', comments='#')
t1 = data[:, 0]  # нс
t2 = data[:, 1]  # тики
t3 = data[:, 2]  # секунды

def calc_stats(values, name):
    print(f"\n--- {name} ---")
    K = len(values)
    t_min = np.min(values)
    t_avg = np.mean(values)
    sigma = np.std(values, ddof=1)
    print(f"K = {K}, min = {t_min:.3f}, avg = {t_avg:.3f}, sigma = {sigma:.3f}")

    # Отбрасывание выбросов по 3 сигма
    mask = np.abs(values - t_avg) <= 3 * sigma
    clean = values[mask]
    if len(clean) < K:
        t_avg_clean = np.mean(clean)
        sigma_clean = np.std(clean, ddof=1)
        print(f"После отбрасывания выбросов: осталось {len(clean)}, avg = {t_avg_clean:.3f}, sigma = {sigma_clean:.3f}")
        # Доверительный интервал для среднего (p=0.95)
        conf = 1.96 * sigma_clean / np.sqrt(len(clean))
        print(f"Доверительный интервал среднего: [{t_avg_clean - conf:.3f}, {t_avg_clean + conf:.3f}]")
    else:
        print("Выбросов не обнаружено")

    # Гистограмма
    plt.figure()
    plt.hist(values, bins=20, alpha=0.5, label='все')
    plt.hist(clean, bins=20, alpha=0.5, label='без выбросов')
    plt.xlabel(name)
    plt.ylabel('Частота')
    plt.legend()
    plt.title(f'Гистограмма {name}')
    plt.savefig(f'../hist_{name.replace(" ","_")}.png')

calc_stats(t1, "clock_gettime (нс)")
calc_stats(t2, "RDTSC (тики)")
calc_stats(t3, "mach_absolute_time (с)")