#include "avx2.h"
#include <immintrin.h>

static void mul_avx2_block(const src_t *a, const src_t *b, dst_t *c)
{
  __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i *>(a));
  __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i *>(b));

  __m128i va_low128 = _mm256_castsi256_si128(va);
  __m128i va_high128 = _mm256_extracti128_si256(va, 1);
  __m128i vb_low128 = _mm256_castsi256_si128(vb);
  __m128i vb_high128 = _mm256_extracti128_si256(vb, 1);

  __m256i va_low = _mm256_cvtepu8_epi16(va_low128);
  __m256i vb_low = _mm256_cvtepu8_epi16(vb_low128);
  __m256i vc_low = _mm256_mullo_epi16(va_low, vb_low);

  __m256i va_high = _mm256_cvtepu8_epi16(va_high128);
  __m256i vb_high = _mm256_cvtepu8_epi16(vb_high128);
  __m256i vc_high = _mm256_mullo_epi16(va_high, vb_high);

  _mm256_store_si256(reinterpret_cast<__m256i *>(c), vc_low);
  _mm256_store_si256(reinterpret_cast<__m256i *>(c + 16), vc_high);
}

void mul_avx2(const src_t *a, const src_t *b, dst_t *c, size_t n)
{
  size_t i = 0;
  const size_t block = 32;
  for (; i + block - 1 < n; i += block)
  {
    mul_avx2_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}

void mul_avx2_unrolled(const src_t *a, const src_t *b, dst_t *c, size_t n, int unroll)
{
  const size_t block = 32;
  size_t i = 0;
  const size_t chunk = unroll * block;
  for (; i + chunk - 1 < n; i += chunk)
  {
    for (int k = 0; k < unroll; ++k)
    {
      mul_avx2_block(a + i + k * block, b + i + k * block, c + i + k * block);
    }
  }
  for (; i + block - 1 < n; i += block)
  {
    mul_avx2_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}