#include "scalar.h"

void mul_scalar(const src_t *a, const src_t *b, dst_t *c, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}

void mul_scalar_unrolled(const src_t *a, const src_t *b, dst_t *c, size_t n, int unroll)
{
  size_t i = 0;
  for (; i + unroll - 1 < n; i += unroll)
  {
    for (int k = 0; k < unroll; ++k)
    {
      c[i + k] = static_cast<dst_t>(a[i + k]) * static_cast<dst_t>(b[i + k]);
    }
  }
  for (; i < n; ++i)
  {
    c[i] = static_cast<dst_t>(a[i]) * static_cast<dst_t>(b[i]);
  }
}