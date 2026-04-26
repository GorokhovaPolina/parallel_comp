#include "sse.h"
#include <emmintrin.h>

static void mul_sse_block(const src_t *a, const src_t *b, dst_t *c)
{
  __m128i va = _mm_load_si128(reinterpret_cast<const __m128i *>(a));
  __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i *>(b));

  __m128i va_low = _mm_unpacklo_epi8(va, _mm_setzero_si128());
  __m128i vb_low = _mm_unpacklo_epi8(vb, _mm_setzero_si128());
  __m128i vc_low = _mm_mullo_epi16(va_low, vb_low);

  __m128i va_high = _mm_unpackhi_epi8(va, _mm_setzero_si128());
  __m128i vb_high = _mm_unpackhi_epi8(vb, _mm_setzero_si128());
  __m128i vc_high = _mm_mullo_epi16(va_high, vb_high);

  _mm_store_si128(reinterpret_cast<__m128i *>(c), vc_low);
  _mm_store_si128(reinterpret_cast<__m128i *>(c + 8), vc_high);
}

void mul_sse(const src_t *a, const src_t *b, dst_t *c, size_t n)
{
  size_t i = 0;
  const size_t block = 16;
  for (; i + block - 1 < n; i += block)
  {
    mul_sse_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}

void mul_sse_unrolled(const src_t *a, const src_t *b, dst_t *c, size_t n, int unroll)
{
  const size_t block = 16;
  size_t i = 0;
  const size_t chunk = unroll * block;
  for (; i + chunk - 1 < n; i += chunk)
  {
    for (int k = 0; k < unroll; ++k)
    {
      mul_sse_block(a + i + k * block, b + i + k * block, c + i + k * block);
    }
  }
  for (; i + block - 1 < n; i += block)
  {
    mul_sse_block(a + i, b + i, c + i);
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}