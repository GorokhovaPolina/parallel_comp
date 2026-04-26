#include "mmx.h"
#include <mmintrin.h>

static void mul_mmx_block(const src_t *a, const src_t *b, dst_t *c)
{
  __m64 va = *reinterpret_cast<const __m64 *>(a);
  __m64 vb = *reinterpret_cast<const __m64 *>(b);

  __m64 va_low = _mm_unpacklo_pi8(va, _mm_setzero_si64());
  __m64 vb_low = _mm_unpacklo_pi8(vb, _mm_setzero_si64());
  __m64 vc_low = _mm_mullo_pi16(va_low, vb_low);

  __m64 va_high = _mm_unpackhi_pi8(va, _mm_setzero_si64());
  __m64 vb_high = _mm_unpackhi_pi8(vb, _mm_setzero_si64());
  __m64 vc_high = _mm_mullo_pi16(va_high, vb_high);

  *reinterpret_cast<__m64 *>(c) = vc_low;
  *reinterpret_cast<__m64 *>(c + 4) = vc_high;
}

void mul_mmx(const src_t *a, const src_t *b, dst_t *c, size_t n)
{
  size_t i = 0;
  const size_t block = 8;
  for (; i + block - 1 < n; i += block)
  {
    mul_mmx_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
  _mm_empty();
}

void mul_mmx_unrolled(const src_t *a, const src_t *b, dst_t *c, size_t n, int unroll)
{
  const size_t block = 8;
  size_t i = 0;
  const size_t chunk = unroll * block;
  for (; i + chunk - 1 < n; i += chunk)
  {
    for (int k = 0; k < unroll; ++k)
    {
      mul_mmx_block(a + i + k * block, b + i + k * block, c + i + k * block);
    }
  }
  for (; i + block - 1 < n; i += block)
  {
    mul_mmx_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
  _mm_empty();
}