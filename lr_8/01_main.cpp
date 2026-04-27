#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

extern "C" void vec_add_cuda(const float *a, const float *b, float *c, int n);

void vec_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    float a[N], b[N], c_gpu[N], c_cpu[N];

    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    vec_add_cuda(a, b, c_gpu, N);
    vec_add_cpu(a, b, c_cpu, N);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(c_gpu[i] - c_cpu[i]) > 1e-5) {
            ok = false;
            std::cout << "Mismatch at " << i << std::endl;
            break;
        }
    }
    if (ok) std::cout << "GPU and CPU results match!\n";

    std::cout << "First 10 sums: ";
    for (int i = 0; i < 10; ++i)
        std::cout << c_gpu[i] << " ";
    std::cout << std::endl;

    return 0;
}
