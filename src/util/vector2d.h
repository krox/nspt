#ifndef UTIL_VECTOR2data_H
#define UTIL_VECTOR2data_H

#include "util/span.h"

namespace util {
template <typename T> class vector2d
{
	std::vector<T> data_;
	size_t height_ = 0;
	size_t width_ = -1;

  public:
	/** constructor */
	vector2d() {}

	/** size metrics */
	bool empty() const { return height_ == 0; }
	size_t height() const { return height_; }
	size_t width() const { return width_; }
	size_t size() const { return height_ * width_; }

	/** row access */
	span<T> operator[](size_t i)
	{
		return span<T>(data_.data() + width_ * i, width_);
	}
	span<const T> operator[](size_t i) const
	{
		return span<const T>(data_.data() + width_ * i, width_);
	}
	span<T> operator()(size_t i)
	{
		return span<T>(data_.data() + width_ * i, width_);
	}
	span<const T> operator()(size_t i) const
	{
		return span<const T>(data_.data() + width_ * i, width_);
	}

	/** element access */
	T &operator()(size_t i, size_t j) { return data_[width_ * i + j]; }
	const T &operator()(size_t i, size_t j) const
	{
		return data_[width_ * i + j];
	}

	/** access as one-dimensional span */
	span<T> flat() { return data_; }
	span<const T> flat() const { return data_; }
	T &flat(size_t i) { return data_[i]; }
	const T &flat(size_t i) const { return data_[i]; }

	/** add/remove row at bottom */
	void push_back(span<const T> v)
	{
		if (width_ == (size_t)-1)
		{
			assert(v.size() > 0);
			assert(data_.size() == 0);
			width_ = v.size();
		}

		assert(v.size() == width_);
		for (size_t i = 0; i < width_; ++i)
			data_.push_back(v[i]);
		height_ += 1;
	}
	void pop_back()
	{
		assert(height_ > 0);
		height_ -= 1;
		data_.resize(width_ * height_);
	}
};

#ifdef NLOHMANN_JSON_HPP

void to_json(nlohmann::json &j, const vector2d<double> &v)
{
	j = nlohmann::json::array();
	for (size_t i = 0; i < v.height(); ++i)
		j.push_back(v[i]);
}

#endif

} // namespace util

#endif
