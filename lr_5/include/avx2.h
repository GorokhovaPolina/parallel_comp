#pragma once
#include "common.h"

void mul_avx2(const src_t *a, const src_t *b, dst_t *c, size_t n);
void mul_avx2_unrolled(const src_t *a, const src_t *b, dst_t *c, size_t n, int unroll);