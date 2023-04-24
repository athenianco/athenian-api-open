#pragma once

#include <stdint.h>
#include <immintrin.h>
#define restrict __restrict__

void sum_repeated_with_step_avx2(
    const int64_t * restrict src,
    int64_t src_len,
    int64_t repeats,
    int64_t step,
    int64_t * restrict dst
) {
    __m256i offset = _mm256_set1_epi64x(0);
    const __m256i step_vec = _mm256_set1_epi64x(step);
	for (int64_t i = 0; i < repeats; i++) {
		int64_t j;
		for (j = 0; j < src_len - 3; j += 4) {
			_mm256_storeu_si256(
				reinterpret_cast<__m256i *>(dst),
				_mm256_add_epi64(
					_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + j)),
					offset
				)
			);
			dst += 4;
		}
		for (; j < src_len; j++) {
			*dst++ = src[j] + step * i;
		}
		offset = _mm256_add_epi64(offset, step_vec);
	}
}