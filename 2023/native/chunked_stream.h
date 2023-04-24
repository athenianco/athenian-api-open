#include <list>
#include "mi_heap_destroy_stl_allocator.h"

template <int chunk_size = (1 << 16)>
class chunked_stream {
 public:
  template <typename T>
  explicit chunked_stream(mi_heap_destroy_stl_allocator<T> &alloc): chunks_(alloc), pos_(0) {
    chunks_.emplace_back(alloc).reserve(chunk_size);
  }

  void write(const void *buffer, size_t size) {
    const char *input = reinterpret_cast<const char *>(buffer);
    int avail = chunk_size - pos_;
	while (size > static_cast<size_t>(avail)) {
	  memcpy(chunks_.back().data() + pos_, input, avail);
	  size -= avail;
	  input += avail;
	  pos_ = 0;
	  chunks_.emplace_back(chunks_.get_allocator()).reserve(chunk_size);
	  avail = chunk_size;
	}
	memcpy(chunks_.back().data() + pos_, input, size);
	pos_ += size;
  }

  size_t dump(char *output, size_t output_size) noexcept {
    size_t total_size = size();
    if (output_size > total_size) {
      output_size = total_size;
    }
    size_t left = output_size;
    auto it = chunks_.begin();
    while (left > chunk_size) {
      memcpy(output, it->data(), chunk_size);
      left -= chunk_size;
      output += chunk_size;
      it++;
    }
    if (left > 0) {
      memcpy(output, it->data(), left);
    }
    return output_size;
  }

  size_t size() const noexcept { return (chunks_.size() - 1) * chunk_size + pos_; }

 private:
  std::list<mi_vector<char>, mi_heap_destroy_stl_allocator<mi_vector<char>>> chunks_;
  int pos_;
};
