#ifndef NSPT_SERIES_H
#define NSPT_SERIES_H

#include <vector>

/**
 * Taylor series: a0 + a1 x + a2 x^2 + ...
 *    - mixing different orders is supported, results are truncated correctly
 *    - T must support operations T+T, T-T, T*T, T*double
 *    - T does not need a default-constructor
 *    - assume operations on T are expansive, so:
 *       - cost of bounds checking and such is negligible
 *       - avoid all unnecessary multiplications/copies
 */
template <typename T> class Series
{
	std::vector<T> a;

  public:
	/** default constructor (zero terms) */
	Series() {}

	/** add term to the back */
	template <typename... Args> void append(Args &&... args)
	{
		a.emplace_back(std::forward<Args...>(args...));
	}

	/** array-like access to terms */
	int size() const { return (int)a.size(); }
	T &operator[](size_t i) { return a.at(i); }
	const T &operator[](size_t i) const { return a.at(i); }
	typename std::vector<T>::iterator begin() { return a.begin(); }
	typename std::vector<T>::iterator end() { return a.end(); }
	typename std::vector<T>::const_iterator begin() const { return a.begin(); }
	typename std::vector<T>::const_iterator end() const { return a.end(); }

	/** set constant term to val, rest to zero */
	void operator=(double val)
	{
		assert(size() >= 1);
		a[0] = val;
		for (int i = 1; i < size(); ++i)
			a[i] = 0.0;
	}

	/** discard all but the first n terms (does nothing if already less) */
	void truncate(int n)
	{
		assert(n >= 0);
		while ((int)a.size() > n)
			a.pop_back();
	}

	/* *this += x^n * b */
	void add_assign(const Series<T> &b, int n, double alpha)
	{
		assert(n >= 0);
		truncate(std::min(size(), b.size() + n));
		for (int i = n; i < size(); ++i)
			a[i] += alpha * b[i - n];
	}
};

template <typename T> Series<T> operator*(const Series<T> &a, double b)
{
	Series<T> r;
	for (int i = 0; i < a.size(); ++i)
		r.append(a[i] * b);
	return r;
}

template <typename T>
Series<T> operator+(const Series<T> &a, const Series<T> &b)
{
	Series<T> r;
	for (int i = 0; i < std::min(a.size(), b.size()); ++i)
		r.append(a[i] + b[i]);
	return r;
}

template <typename T>
Series<T> operator-(const Series<T> &a, const Series<T> &b)
{
	Series<T> r;
	for (int i = 0; i < std::min(a.size(), b.size()); ++i)
		r.append(a[i] - b[i]);
	return r;
}

template <typename T>
Series<T> operator*(const Series<T> &a, const Series<T> &b)
{
	Series<T> r;
	for (int i = 0; i < std::min(a.size(), b.size()); ++i)
	{
		r.append(a[0] * b[i]);
		for (int j = 1; j <= i; ++j)
			r[i] += a[j] * b[i - j];
	}
	return r;
}

template <typename T> void operator*=(Series<T> &a, double b)
{
	for (int i = 0; i < a.size(); ++i)
		a[i] *= b;
}

template <typename T> void operator+=(Series<T> &a, const Series<T> &b)
{
	a.truncate(std::min(a.size(), b.size()));
	for (int i = 0; i < a.size(); ++i)
		a[i] += b[i];
}

template <typename T> void operator-=(Series<T> &a, const Series<T> &b)
{
	a.truncate(std::min(a.size(), b.size()));
	for (int i = 0; i < a.size(); ++i)
		a[i] -= b[i];
}

inline double norm2(double x) { return x * x; }

/** only implemented for a[0] == 0 */
template <typename T> Series<T> exp(const Series<T> &a_)
{
	assert(a_.size() >= 1);
	assert(norm2(a_[0]) < 1.0e-10);

	Series<T> a;
	for (int i = 1; i < a_.size(); ++i)
		a.append(a_[i]);
	Series<T> an = a;
	Series<T> r = a_;
	r[0] = 1.0;

	double f = 1.0;
	for (int i = 2; i < a_.size(); ++i)
	{
		a.truncate(a.size() - 1);
		an = an * a;
		f *= 1.0 / i;
		r.add_assign(an, i, f);
	}
	return r;
}

/** only implemented for a[0] == 1 */
template <typename T> Series<T> log(const Series<T> &a_)
{
	assert(a_.size() >= 1);
	assert(norm2(T(a_[0] - 1.0)) < 1.0e-10);

	Series<T> a;
	for (int i = 1; i < a_.size(); ++i)
		a.append(a_[i]);
	Series<T> an = a;
	Series<T> r = a_;
	r[0] = 0.0;

	for (int i = 2; i < a_.size(); ++i)
	{
		a.truncate(a.size() - 1);
		an = an * a;
		r.add_assign(an, i, i % 2 ? 1.0 / i : -1.0 / i);
	}
	return r;
}

template <typename T>
Series<Lattice<T>> Cshift(const Series<Lattice<T>> &a, int mu, int dist)
{
	Series<Lattice<T>> r;
	for (int i = 0; i < a.size(); ++i)
		r.append(Grid::Cshift(a[i], mu, dist));
	return r;
}

template <typename T> Series<Lattice<T>> adj(const Series<Lattice<T>> &a)
{
	Series<Lattice<T>> r;
	for (int i = 0; i < a.size(); ++i)
		r.append(Grid::adj(a[i]));
	return r;
}

template <typename T> Series<Lattice<T>> Ta(const Series<Lattice<T>> &a)
{
	Series<Lattice<T>> r;
	for (int i = 0; i < a.size(); ++i)
		r.append(Grid::Ta(a[i]));
	return r;
}

/*template <typename T>
Series<Lattice<typename T::scalar_type>> trace(const Series<Lattice<T>> &a)
{
    Series<typename T::scalar_type> r;
    for (int i = 0; i < a.size(); ++i)
        r.append(Grid::trace(a[i]));
    return r;
}

template <typename T> Series<T> sum(const Series<Lattice<T>> &a)
{
    Series<T> r;
    for (int i = 0; i < a.size(); ++i)
        r.append(Grid::sum(a[i]));
    return r;
}*/

#endif
