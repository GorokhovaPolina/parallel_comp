#include <windows.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <intrin.h> 

using namespace std;

const int N = 1000;

// Сортировка пузырьком – измеряемый фрагмент
void bubbleSort() {
    int arr[N];
    for (int i = 0; i < N; i++) arr[i] = rand() % 1000;
    bool changed;
    do {
        changed = false;
        for (int i = 0; i < N-1; i++) {
            if (arr[i] > arr[i+1]) {
                int t = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = t;
                changed = true;
            }
        }
    } while (changed);
}

int main() {
    // Повышаем приоритет процесса, чтобы уменьшить влияние других задач
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    // Привязываем поток к первому ядру (маска 1) – для стабильности TSC
    SetThreadAffinityMask(GetCurrentThread(), 1);

    const int K = 100;   // число замеров (можно изменить по заданию)

    // Векторы для хранения результатов
    vector<DWORD> t1(K);                // GetTickCount (мс)
    vector<LONGLONG> t2(K);             // QueryPerformanceCounter (тики)
    vector<unsigned __int64> t3(K);     // RDTSC (такты процессора)

    // Частота QPC (нужна только если захотите перевести в секунды)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    cout << "Измерение " << K << " раз..." << endl;

    for (int i = 0; i < K; i++) {
        // Способ 1: GetTickCount (низкая точность, миллисекунды)
        DWORD start1 = GetTickCount();
        bubbleSort();
        t1[i] = GetTickCount() - start1;

        // Способ 2: QueryPerformanceCounter (высокая точность)
        LARGE_INTEGER start2, end2;
        QueryPerformanceCounter(&start2);
        bubbleSort();
        QueryPerformanceCounter(&end2);
        t2[i] = end2.QuadPart - start2.QuadPart;

        // Способ 3: RDTSC (счётчик тактов процессора)
        unsigned __int64 start3 = __rdtsc();
        bubbleSort();
        t3[i] = __rdtsc() - start3;

        // Прогресс
        if ((i + 1) % 10 == 0)
            cout << "Выполнено " << i + 1 << " замеров" << endl;
    }

    // Сохраняем результаты в один файл (три столбца)
    ofstream out("data/data_windows.txt");
    out << "# GetTickCount_ms QPC_ticks RDTSC_ticks\n";
    out << fixed << setprecision(0);   // целые числа
    for (int i = 0; i < K; i++) {
        out << t1[i] << " " << t2[i] << " " << t3[i] << "\n";
    }
    out.close();

    cout << "Результаты сохранены в data_windows.txt" << endl;
    return 0;
}