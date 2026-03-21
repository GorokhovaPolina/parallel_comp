#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>

// ======================== Вспомогательные функции ========================

// Заполнение матрицы случайными числами от 0 до 1
void fill_random(float* mat, int N) {
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Сравнение двух матриц с заданной точностью
bool compare_matrices(const float* C1, const float* C2, int N, float eps = 1e-5f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(C1[i] - C2[i]) > eps) return false;
    }
    return true;
}

// Печать маленькой матрицы (для отладки)
void print_matrix(const float* mat, int N, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < std::min(N, 8); ++i) {
        for (int j = 0; j < std::min(N, 8); ++j)
            std::cout << std::fixed << std::setprecision(2) << mat[i * N + j] << " ";
        std::cout << (N > 8 ? "..." : "") << "\n";
    }
    if (N > 8) std::cout << "...\n";
}

// Измерение времени выполнения функции в секундах
double measure_time(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

// Вычисление GFLOP/s
double compute_gflops(int N, double time_sec) {
    return (2.0 * N * N * N) / (time_sec * 1e9);
}

// ======================== Алгоритмы умножения ========================

// 1. Классическое умножение (i,j,k)
void mul_classic(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. С предварительным транспонированием B
void mul_transposed(const float* A, const float* B, float* C, int N) {
    // Транспонирование B
    float* BT = new float[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            BT[j * N + i] = B[i * N + j];

    // Умножение
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * BT[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
    delete[] BT;
}

// 3. Буферизованное умножение (копирование столбца) с раскруткой M
void mul_buffered_unrolled(const float* A, const float* B, float* C, int N, int M) {
    float* tmp = new float[N];
    for (int j = 0; j < N; ++j) {
        // Копируем j-й столбец B
        for (int k = 0; k < N; ++k)
            tmp[k] = B[k * N + j];

        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            int k = 0;
            // Раскрутка по M итераций
            for (; k + M <= N; k += M) {
                for (int t = 0; t < M; ++t) {
                    sum += A[i * N + k + t] * tmp[k + t];
                }
            }
            // Остаток
            for (; k < N; ++k) {
                sum += A[i * N + k] * tmp[k];
            }
            C[i * N + j] = sum;
        }
    }
    delete[] tmp;
}

// 4. Блочное умножение с раскруткой внутреннего цикла по j (M итераций)
void mul_blocked_unrolled(const float* A, const float* B, float* C, int N, int S, int M) {
    // Обнуляем C
    std::fill(C, C + N * N, 0.0f);

    for (int iBlock = 0; iBlock < N; iBlock += S) {
        int iMax = std::min(iBlock + S, N);
        for (int jBlock = 0; jBlock < N; jBlock += S) {
            int jMax = std::min(jBlock + S, N);
            for (int kBlock = 0; kBlock < N; kBlock += S) {
                int kMax = std::min(kBlock + S, N);
                // Умножение блоков
                for (int i = iBlock; i < iMax; ++i) {
                    for (int k = kBlock; k < kMax; ++k) {
                        float a = A[i * N + k];
                        int j = jBlock;
                        // Раскрутка по j на M итераций
                        for (; j + M <= jMax; j += M) {
                            for (int t = 0; t < M; ++t) {
                                C[i * N + j + t] += a * B[k * N + j + t];
                            }
                        }
                        // Остаток
                        for (; j < jMax; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ======================== Эксперименты ========================

// Проверка корректности всех алгоритмов (для маленького N)
void test_correctness(int N = 32) {
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C1 = new float[N * N]();
    float* C2 = new float[N * N]();
    fill_random(A, N);
    fill_random(B, N);

    mul_classic(A, B, C1, N);
    bool ok = true;

    // Транспонированный
    mul_transposed(A, B, C2, N);
    if (!compare_matrices(C1, C2, N)) {
        std::cerr << "Error: transposed multiplication mismatch\n";
        ok = false;
    }

    // Буферизованный (M=1)
    mul_buffered_unrolled(A, B, C2, N, 1);
    if (!compare_matrices(C1, C2, N)) {
        std::cerr << "Error: buffered multiplication mismatch\n";
        ok = false;
    }

    // Блочный (S=16, M=1)
    mul_blocked_unrolled(A, B, C2, N, 16, 1);
    if (!compare_matrices(C1, C2, N)) {
        std::cerr << "Error: blocked multiplication mismatch\n";
        ok = false;
    }

    if (ok)
        std::cout << "All algorithms produce correct results for N=" << N << "\n";

    delete[] A; delete[] B; delete[] C1; delete[] C2;
}

// Измерение производительности для заданного алгоритма
void measure_and_print(const std::string& name, 
                       std::function<void()> func, 
                       int N, int repeat = 3) {
    double total_time = 0.0;
    for (int r = 0; r < repeat; ++r) {
        total_time += measure_time(func);
    }
    double avg_time = total_time / repeat;
    double gflops = compute_gflops(N, avg_time);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left << std::setw(30) << name
              << " | N=" << std::setw(6) << N
              << " | time=" << std::setw(8) << avg_time << " s"
              << " | GFLOPS=" << std::setw(8) << gflops << "\n";
}

// Основная программа
int main() {
    // Инициализация генератора случайных чисел
    srand(42);

    // ==================== 1. Проверка корректности ====================
    std::cout << "=== Correctness check ===\n";
    test_correctness(64);
    std::cout << "\n";

    // ==================== 2. Debug vs Release ====================
    // Для этого нужно скомпилировать два раза: без оптимизаций и с оптимизациями
    // Здесь просто показываем, как будет выглядеть вывод в Release.
    // При запуске в Debug (без -O3) будет медленнее.
    std::cout << "=== Performance comparison (Release build) ===\n";
    std::cout << "Algorithm                      | N      | time (s) | GFLOPS   \n";
    std::cout << "-------------------------------------------------------------\n";

    // Выбираем N для экспериментов (можно менять)
    // Для быстрых тестов - небольшие N, для полной картины - от 32 до 2048
    // Рекомендуемые N: 32, 64, 128, 256, 512, 1024, 2048, 4096 (если хватает памяти)
    int test_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};

    for (int N : test_sizes) {
        // Выделяем память
        float* A = new float[N * N];
        float* B = new float[N * N];
        float* C = new float[N * N]();
        fill_random(A, N);
        fill_random(B, N);

        // 1. Классическое умножение
        measure_and_print("Classic (ijk)", [&]() { mul_classic(A, B, C, N); }, N);

        // 2. Транспонирование
        measure_and_print("Transposed B", [&]() { mul_transposed(A, B, C, N); }, N);

        // 3. Буферизованное с оптимальной раскруткой (найдём позже)
        // Для предварительной оценки используем M=4
        measure_and_print("Buffered (M=4)", [&]() { mul_buffered_unrolled(A, B, C, N, 4); }, N);

        // 4. Блочное с оптимальными параметрами (S=64, M=4)
        measure_and_print("Blocked (S=64, M=4)", [&]() { mul_blocked_unrolled(A, B, C, N, 64, 4); }, N);

        std::cout << "-------------------------------------------------------------\n";

        delete[] A; delete[] B; delete[] C;
    }

    // ==================== 3. Поиск оптимальной раскрутки M для буферизованного ====================
    std::cout << "\n=== Buffered multiplication: varying unroll factor M ===\n";
    int N_fixed = 1024;  // фиксируем N (можно выбрать другой)
    float* A = new float[N_fixed * N_fixed];
    float* B = new float[N_fixed * N_fixed];
    float* C = new float[N_fixed * N_fixed]();
    fill_random(A, N_fixed);
    fill_random(B, N_fixed);

    std::cout << "N = " << N_fixed << "\n";
    std::cout << "M       time (s)   GFLOPS\n";
    for (int M = 1; M <= 16; M *= 2) {
        double t = measure_time([&]() { mul_buffered_unrolled(A, B, C, N_fixed, M); });
        double gflops = compute_gflops(N_fixed, t);
        std::cout << std::setw(2) << M << "       " 
                  << std::fixed << std::setprecision(4) << t << "     " << gflops << "\n";
    }
    delete[] A; delete[] B; delete[] C;

    // ==================== 4. Поиск оптимального размера блока S и раскрутки M для блочного ====================
    std::cout << "\n=== Blocked multiplication: varying block size S and unroll M ===\n";
    N_fixed = 1024;
    A = new float[N_fixed * N_fixed];
    B = new float[N_fixed * N_fixed];
    C = new float[N_fixed * N_fixed]();
    fill_random(A, N_fixed);
    fill_random(B, N_fixed);

    std::cout << "N = " << N_fixed << "\n";
    std::cout << "S\tM=1\tM=2\tM=4\tM=8\tM=16 (GFLOPS)\n";
    for (int S = 1; S <= 256; S *= 2) {
        std::cout << std::setw(3) << S << "\t";
        for (int M = 1; M <= 16; M *= 2) {
            double t = measure_time([&]() { mul_blocked_unrolled(A, B, C, N_fixed, S, M); });
            double gflops = compute_gflops(N_fixed, t);
            std::cout << std::fixed << std::setprecision(2) << gflops << "\t";
        }
        std::cout << "\n";
    }
    delete[] A; delete[] B; delete[] C;

    std::cout << "\n=== Done. Use these results to build graphs and draw conclusions. ===\n";
    return 0;
}