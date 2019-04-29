#ifndef UTIL_SPAN_H
#define UTIL_SPAN_H

#include <cassert>
#include <type_traits>
#include <vector>

namespace util {

/** contiguous 1D array-view */
template <typename T> class span
{
	T *data_ = nullptr;
	size_t size_ = 0;

	typedef typename std::remove_cv<T>::type T_mut;

  public:
	using value_type = T;
	using size_type = size_t;
	using index_type = size_t;
	using iterator = T *;
	using const_iterator = const T *;

	/** constructors */
	span() = default;
	span(T *data, size_t size) : data_(data), size_(size) {}
	span(T *begin, T *end) : data_(begin), size_(end - begin) {}
	span(std::vector<T> &v) : data_(v.data()), size_(v.size()) {}
	span(const std::vector<T_mut> &v) : data_(v.data()), size_(v.size()) {}
	span(const span<T_mut> &v) : data_(v.data()), size_(v.size()) {}

	/** field access */
	T *data() { return data_; }
	const T *data() const { return data_; }
	size_t size() const { return size_; }

	/** element access */
	T &operator[](size_t i) { return data_[i]; }
	const T &operator[](size_t i) const { return data_[i]; }
	T &operator()(int i) { return data_[i]; }
	const T &operator()(int i) const { return data_[i]; }

	/** iterators */
	iterator begin() { return data_; }
	iterator end() { return data_ + size_; }
	const_iterator begin() const { return data_; }
	const_iterator end() const { return data_ + size_; }
	const_iterator cbegin() const { return data_; }
	const_iterator cend() const { return data_ + size_; }

	/** supspan */
	span<T> subspan(size_t a, size_t b)
	{
		return span<T>(data_ + a, data_ + b);
	}
};

/** strided 1D array-view */
template <typename T> class gspan
{
	T *data_ = nullptr;
	size_t size_ = 0;
	size_t stride_ = 1;

	typedef typename std::remove_cv<T>::type T_mut;

  public:
	using value_type = T;
	using size_type = size_t;
	using index_type = size_t;
	// NOTE: don't use T* as iterator

	/** constructors */
	gspan() = default;
	gspan(T *data, size_t size, size_t stride)
	    : data_(data), size_(size), stride_(stride)
	{}
	gspan(std::vector<T> &v) : data_(v.data()), size_(v.size()) {}
	gspan(const std::vector<T_mut> &v) : data_(v.data()), size_(v.size()) {}
	gspan(span<T> &v) : data_(v.data()), size_(v.size()) {}
	gspan(const gspan<T_mut> &v)
	    : data_(v.data()), size_(v.size()), stride_(v.stride())
	{}

	/** field access */
	T *data() { return data_; }
	const T *data() const { return data_; }
	size_t size() const { return size_; }
	size_t stride() const { return stride_; }

	/** element access */
	T &operator[](size_t i) { return data_[i * stride_]; }
	const T &operator[](size_t i) const { return data_[i * stride_]; }
	T &operator()(int i) { return data_[i * stride_]; }
	const T &operator()(int i) const { return data_[i * stride_]; }

	/** supspan */
	gspan<T> subspan(size_t a, size_t b)
	{
		assert(a <= b && b <= size_);
		return gspan<T>(data_ + a * stride_, b - a, stride_);
	}
};

#ifdef NLOHMANN_JSON_HPP

inline void to_json(nlohmann::json &j, span<const double> v)
{
	j = nlohmann::json::array();
	for (size_t i = 0; i < v.size(); ++i)
		j.push_back(v[i]);
}

#endif

} // namespace util

#endif
