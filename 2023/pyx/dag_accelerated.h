#pragma once
#include <immintrin.h>

size_t sorted_set_difference_avx2(
	const uint32_t *__restrict__ set1,
	const size_t length1,
	const uint32_t *__restrict__ set2,
    const size_t length2,
    uint32_t *__restrict__ out) {
  __m256i left, right;
  const __m256i ones = _mm256_set1_epi8(0xff);
  size_t passed = 0;
  const uint32_t *border_left = set1 + length1 - 8;
  const uint32_t *border_right = set2 + length2 - 8;
  while (set1 <= border_left && set2 <= border_right) {
    left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(set1));
    right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(set2));
    __m256i c = _mm256_cmpeq_epi32(left, right);
    if (_mm256_testc_si256(c, ones)) {
    	set1 += 8;
    	set2 += 8;
    	continue;
    }
    int offset = __builtin_ctz(~static_cast<unsigned>(_mm256_movemask_epi8(c))) >> 2;
    set1 += offset;
    set2 += offset;
    if (*set1 < *set2) {
      out[passed++] = *set1++;
    } else {
      set2++;
    }
  }
  border_left += 8;
  border_right += 8;
  while (set1 < border_left && set2 < border_right) {
    uint32_t ileft = *set1;
    uint32_t iright = *set2;
    if (ileft == iright) {
      set1++;
      set2++;
    } else if (ileft < iright) {
      out[passed++] = ileft;
      set1++;
    } else {
      set2++;
    }
  }
  if (set2 == border_right) {
    while (set1 < border_left) {
      out[passed++] = *set1++;
    }
  }
  return passed;
}