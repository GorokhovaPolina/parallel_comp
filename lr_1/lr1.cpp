#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <mach/mach_time.h>
#include <unistd.h>
#include <time.h>
#include <x86intrin.h>
#include <sys/resource.h> // для повышения приоритета
#include <fstream>

using namespace std;

void bubbleSort()
{
    const int N = 1000;
    int arr[N];
    for (int i = 0; i < N; i++)
        arr[i] = rand() % 1000;

    bool changed;
    do
    {
        changed = false;
        for (int i = 0; i < N - 1; i++)
        {
            if (arr[i] > arr[i + 1])
            {
                int t = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = t;
                changed = true;
            }
        }
    } while (changed);
}

// Способ 1: clock_gettime (нс)
uint64_t getTimeNanos()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Способ 2: RDTSC (для Intel)
uint64_t getHighPrecisionCounter()
{
    return __rdtsc();
}

// Способ 3: mach_absolute_time с преобразованием в сек
double getAbsoluteTimeSeconds()
{
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0)
        mach_timebase_info(&info);
    uint64_t time = mach_absolute_time();
    return (double)time * (double)info.numer / (double)info.denom / 1e9;
}

int main()
{
    setpriority(PRIO_PROCESS, 0, -20);

    const int K = 100; // число замеров

    vector<uint64_t> t1(K); // clock_gettime (нс)
    vector<uint64_t> t2(K); // RDTSC (тики)
    vector<double> t3(K);   // mach_absolute_time (сек)

    cout << "Измерение " << K << " раз..." << endl;

    for (int i = 0; i < K; i++)
    {
        // Способ 1
        uint64_t start1 = getTimeNanos();
        bubbleSort();
        uint64_t end1 = getTimeNanos();
        t1[i] = end1 - start1;

        // Способ 2
        uint64_t start2 = getHighPrecisionCounter();
        bubbleSort();
        uint64_t end2 = getHighPrecisionCounter();
        t2[i] = end2 - start2;

        // Способ 3
        double start3 = getAbsoluteTimeSeconds();
        bubbleSort();
        double end3 = getAbsoluteTimeSeconds();
        t3[i] = end3 - start3;

        cout << "Замер " << i + 1 << ": "
             << t1[i] << " нс, "
             << t2[i] << " тиков, "
             << t3[i] << " с" << endl;
    }

    ofstream out("./data.txt");
    out << "# clock_gettime_ns rdtsc_ticks mach_abs_s\n";
    out << fixed << setprecision(9);  // для третьего столбца
    for (int i = 0; i < K; i++) {
        out << t1[i] << " " << t2[i] << " " << t3[i] << "\n";
    }
    out.close();
    return 0;
}