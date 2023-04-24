#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mimalloc.h>

template<
    class T,
    class U,
    class HASH = std::hash<T>,
    class PRED = std::equal_to<T>
>
using mi_unordered_map = std::unordered_map<T, U, HASH, PRED, mi_heap_destroy_stl_allocator<std::pair<const T, U>>>;

template<
    class T,
    class HASH = std::hash<T>,
    class PRED = std::equal_to<T>
>
using mi_unordered_set = std::unordered_set<T, HASH, PRED, mi_heap_destroy_stl_allocator<T>>;

template<class T>
using mi_vector = std::vector<T, mi_heap_destroy_stl_allocator<T>>;

using mi_string = std::basic_string<char, std::char_traits<char>, mi_heap_destroy_stl_allocator<char>>;

namespace std {
    template<> struct hash<mi_string> {
        size_t operator()(const mi_string &s) const {
            return std::hash<std::string_view>()(s);
        }
    };
}

struct empty_deleter {
  template <typename T>
  void operator()(T *) const noexcept {}
};
