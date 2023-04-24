#include <type_traits>

#include <immintrin.h>

#define interleave_lo do { \
  __m256i tlo = _mm256_permute4x64_epi64(s, 0b00010100); \
  tlo = _mm256_unpacklo_epi8(tlo, zeros); \
  if (step == 2) { \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step), tlo); \
  } else { \
	__m256i tlolo = _mm256_permute4x64_epi64(tlo, 0b00010100); \
	tlolo = _mm256_unpacklo_epi8(tlolo, zeros); \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step), tlolo); \
	__m256i tlohi = _mm256_permute4x64_epi64(tlo, 0b11101011); \
	tlohi = _mm256_unpackhi_epi8(tlohi, zeros); \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step + 32), tlohi); \
  } \
} while (false)

#define interleave_hi do { \
  __m256i thi = _mm256_permute4x64_epi64(s, 0b11101011); \
  thi = _mm256_unpackhi_epi8(thi, zeros); \
  if (step == 2) { \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step + 32), thi); \
  } else { \
	__m256i thilo = _mm256_permute4x64_epi64(thi, 0b00010100); \
	thilo = _mm256_unpacklo_epi8(thilo, zeros); \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step + 64), thilo); \
	__m256i thihi = _mm256_permute4x64_epi64(thi, 0b11101011); \
	thihi = _mm256_unpackhi_epi8(thihi, zeros); \
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * step + 96), thihi); \
  } \
} while (false)

template <int step, typename = std::enable_if_t<step == 2 || step == 4>>
void interleave_bytes(const char *__restrict__ src, ssize_t length, char *__restrict__ out) {
  const __m256i zeros = _mm256_setzero_si256();
  __m256i s;
  ssize_t i;
  for (i = 0; i < length - 31; i += 32) {
    s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
    interleave_lo;
    interleave_hi;
  }
  if (i < length - 15) {
	const __m128i *head = reinterpret_cast<const __m128i *>(src + i);
	s = _mm256_loadu2_m128i(head, head);
	interleave_lo;
	i += 16;
  }
  if (step == 2) {
	for (; i < length - 3; i += 4) {
	  uint64_t quad = *reinterpret_cast<const uint32_t *>(src + i);
	  *reinterpret_cast<uint64_t *>(out + i * step) = ((quad & 0xFF000000) << 24) | ((quad & 0xFF0000) << 16) | ((quad & 0xFF00) << 8) | (quad & 0xFF);
	}
	for (; i < length; i++) {
	  *reinterpret_cast<uint16_t *>(out + i * step) = static_cast<const uint16_t>(reinterpret_cast<const uint8_t *>(src)[i]);
	}
  } else {
	for (; i < length - 1; i += 2) {
	  uint64_t pair = *reinterpret_cast<const uint16_t *>(src + i);
	  *reinterpret_cast<uint64_t *>(out + i * step) = ((pair & 0xFF00) << 24) | (pair & 0xFF);
	}
	if (i < length) {
	  *reinterpret_cast<uint32_t *>(out + i * step) = static_cast<const uint32_t>(reinterpret_cast<const uint8_t *>(src)[i]);
	}
  }
}

constexpr auto interleave_bytes2 = interleave_bytes<2>;
constexpr auto interleave_bytes4 = interleave_bytes<4>;


void interleave_bytes24(const char *__restrict__ src, ssize_t length, char *__restrict__ out) {
  const __m256i zeros = _mm256_setzero_si256();
  __m256i s;
  ssize_t i;
  for (i = 0; i < length - 31; i += 32) {
    s = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
    __m256i tlo = _mm256_permute4x64_epi64(s, 0b00010100);
  	tlo = _mm256_unpacklo_epi16(tlo, zeros);
  	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * 2), tlo);
  	__m256i thi = _mm256_permute4x64_epi64(s, 0b11101011);
  	thi = _mm256_unpackhi_epi16(thi, zeros);
  	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * 2 + 32), thi);
  }
  if (i < length - 15) {
	const __m128i *head = reinterpret_cast<const __m128i *>(src + i);
	s = _mm256_loadu2_m128i(head, head);
	__m256i tlo = _mm256_permute4x64_epi64(s, 0b00010100);
  	tlo = _mm256_unpacklo_epi16(tlo, zeros);
  	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i * 2), tlo);
	i += 16;
  }
  for (; i < length - 3; i += 4) {
  	uint64_t pair = *reinterpret_cast<const uint32_t *>(src + i);
	*reinterpret_cast<uint64_t *>(out + i * 2) = (pair & 0xFFFF) | ((pair & 0xFFFF0000) << 16);
  }
  if (i < length) {
    *reinterpret_cast<uint32_t *>(out + i * 2) = static_cast<const uint32_t>(*reinterpret_cast<const uint16_t *>(src + i));
  }
}