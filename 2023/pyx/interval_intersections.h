#include <algorithm>

template <typename T, typename I>
void argsort_bodies(const mi_vector<T> &bodies, mi_vector<I> &indexes) noexcept {
  std::sort(indexes.begin(), indexes.end(), [&bodies](I left, I right) -> bool {
	  return bodies[left] < bodies[right];
  });
}