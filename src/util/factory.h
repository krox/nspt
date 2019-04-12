#ifndef UTIL_FACTORY_H
#define UTIL_FACTORY_H

#include <map>
#include <string>
#include <typeinfo>

#include <fmt/format.h>

namespace util {

class Object
{
  public:
	const size_t th_; // typeid(T).hash_code()
	Object(size_t th) : th_(th){};
	virtual ~Object(){};
};

template <typename T> class Holder : public Object
{
  public:
	std::unique_ptr<T> ptr_;
	Holder(T *ptr) : Object(typeid(T).hash_code()), ptr_(ptr) {}
	virtual ~Holder() {}
};

class Store
{
	std::map<std::string, std::unique_ptr<Object>> table;

  public:
	template <typename T, typename... Ts>
	T &create(const std::string &name, Ts &&... args)
	{
		if (table.count(name) != 0)
			throw std::range_error(
			    fmt::format("Named object '{}' already exists.", name));

		T *ptr = new T{std::forward<Ts>(args)...};
		table[name] = std::make_unique<Holder<T>>(ptr);
		return *ptr;
	}

	template <typename T> T &get(const std::string &name)
	{
		if (table.count(name) == 0)
			throw std::range_error(
			    fmt::format("Named object '{}' not found.", name));

		Object &obj = *table[name];
		if (obj.th_ != typeid(T).hash_code())
			throw std::range_error(
			    fmt::format("Named object '{}' has wrong type.", name));

		Holder<T> &h = dynamic_cast<Holder<T> &>(obj);
		return *h.ptr_;
	}
};

} // namespace util
#endif
