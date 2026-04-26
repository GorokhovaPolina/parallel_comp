#pragma once
#include "common.h"

void mul_mmx(const src_t* a, const src_t* b, dst_t* c, size_t n);
void mul_mmx_unrolled(const src_t* a, const src_t* b, dst_t* c, size_t n, int unroll);