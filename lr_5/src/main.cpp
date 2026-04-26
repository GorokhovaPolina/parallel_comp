#include "common.h"
#include "scalar.h"
#include "mmx.h"
#include "sse.h"
#include "avx2.h"

#include <random>
#include <iomanip>

void init_data(src_t *a, src_t *b, size_t n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<src_t> dist(0, 255);
  for (size_t i = 0; i < n; ++i)
  {
    a[i] = dist(gen);
    b[i] = dist(gen);
  }
}

int main()
{
  const size_t n = N;
  src_t *a = static_cast<src_t *>(aligned_alloc_(32, n * sizeof(src_t)));
  src_t *b = static_cast<src_t *>(aligned_alloc_(32, n * sizeof(src_t)));
  dst_t *ref = static_cast<dst_t *>(aligned_alloc_(32, n * sizeof(dst_t)));
  dst_t *res = static_cast<dst_t *>(aligned_alloc_(32, n * sizeof(dst_t)));

  if (!a || !b || !ref || !res)
  {
    std::cerr << "Memory allocation failed\n";
    return 1;
  }

  init_data(a, b, n);
  mul_scalar(a, b, ref, n);

  std::cout << "\n=== Multiplication uint8_t -> uint16_t, N = " << n << "\n\n";
  std::cout << std::left << std::setw(22) << "Implementation"
            << std::right << std::setw(12) << "Time (ms)"
            << std::setw(12) << "Speedup" << std::endl;
  std::cout << std::string(46, '-') << std::endl;

  // Scalar
  Timer t;
  mul_scalar(a, b, res, n);
  double t_scalar = t.elapsed_ms();
  std::cout << std::left << std::setw(22) << "Scalar (O2)"
            << std::right << std::setw(12) << std::fixed << std::setprecision(3) << t_scalar
            << std::setw(12) << "1.00" << std::endl;
  if (!verify(ref, res, n))
    std::cout << "  [FAIL]\n";

  for (int unroll : {2, 4, 8})
  {
    t.reset();
    mul_scalar_unrolled(a, b, res, n, unroll);
    double time = t.elapsed_ms();
    std::cout << std::left << std::setw(22) << ("Scalar unroll " + std::to_string(unroll))
              << std::right << std::setw(12) << time
              << std::setw(12) << (t_scalar / time) << std::endl;
    if (!verify(ref, res, n))
      std::cout << "  [FAIL]\n";
  }

  // MMX
  t.reset();
  mul_mmx(a, b, res, n);
  double t_mmx = t.elapsed_ms();
  std::cout << std::left << std::setw(22) << "MMX"
            << std::right << std::setw(12) << t_mmx
            << std::setw(12) << (t_scalar / t_mmx) << std::endl;
  if (!verify(ref, res, n))
    std::cout << "  [FAIL]\n";

  for (int unroll : {2, 4, 8})
  {
    t.reset();
    mul_mmx_unrolled(a, b, res, n, unroll);
    double time = t.elapsed_ms();
    std::cout << std::left << std::setw(22) << ("MMX unroll " + std::to_string(unroll))
              << std::right << std::setw(12) << time
              << std::setw(12) << (t_scalar / time) << std::endl;
    if (!verify(ref, res, n))
      std::cout << "  [FAIL]\n";
  }

  // SSE2
  t.reset();
  mul_sse(a, b, res, n);
  double t_sse = t.elapsed_ms();
  std::cout << std::left << std::setw(22) << "SSE2"
            << std::right << std::setw(12) << t_sse
            << std::setw(12) << (t_scalar / t_sse) << std::endl;
  if (!verify(ref, res, n))
    std::cout << "  [FAIL]\n";

  for (int unroll : {2, 4, 8})
  {
    t.reset();
    mul_sse_unrolled(a, b, res, n, unroll);
    double time = t.elapsed_ms();
    std::cout << std::left << std::setw(22) << ("SSE2 unroll " + std::to_string(unroll))
              << std::right << std::setw(12) << time
              << std::setw(12) << (t_scalar / time) << std::endl;
    if (!verify(ref, res, n))
      std::cout << "  [FAIL]\n";
  }

  // AVX2
  t.reset();
  mul_avx2(a, b, res, n);
  double t_avx = t.elapsed_ms();
  std::cout << std::left << std::setw(22) << "AVX2"
            << std::right << std::setw(12) << t_avx
            << std::setw(12) << (t_scalar / t_avx) << std::endl;
  if (!verify(ref, res, n))
    std::cout << "  [FAIL]\n";

  for (int unroll : {2, 4, 8})
  {
    t.reset();
    mul_avx2_unrolled(a, b, res, n, unroll);
    double time = t.elapsed_ms();
    std::cout << std::left << std::setw(22) << ("AVX2 unroll " + std::to_string(unroll))
              << std::right << std::setw(12) << time
              << std::setw(12) << (t_scalar / time) << std::endl;
    if (!verify(ref, res, n))
      std::cout << "  [FAIL]\n";
  }

  aligned_free_(a);
  aligned_free_(b);
  aligned_free_(ref);
  aligned_free_(res);
  return 0;
}