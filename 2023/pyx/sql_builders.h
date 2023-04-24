#include <immintrin.h>

static inline int scan_unicode_kind(const char *data, long length) {
	__m256i word = _mm256_set1_epi8(0);
	long i;
	for (i = 0; i < length - 31; i += 32) {
		word = _mm256_or_si256(word, _mm256_loadu_si256((const __m256i *)(data + i)));
	}
	word = _mm256_cmpeq_epi8(word, _mm256_set1_epi8(0));
	uint32_t mask = _mm256_movemask_epi8(word);
	mask = ~mask;
	for (; i < length; i += 4) {
	    mask |= data[i] != 0;
	    mask |= (data[i + 1] != 0) << 1;
	    mask |= (data[i + 2] != 0) << 2;
	    mask |= (data[i + 3] != 0) << 3;
	}
	if (mask & ((1 << 3) | (1 << 7) | (1 << 11) | (1 << 15) | (1 << 19) | (1 << 23) | (1 << 27) | (1 << 31))) {
	    return 4;
	}
	if (mask & ((1 << 2) | (1 << 6) | (1 << 10) | (1 << 14) | (1 << 18) | (1 << 22) | (1 << 26) | (1 << 30))) {
	    return 4;
	}
	if (mask & ((1 << 1) | (1 << 5) | (1 << 9) | (1 << 13) | (1 << 17) | (1 << 21) | (1 << 25) | (1 << 29))) {
	    return 2;
	}
	return 1;
}