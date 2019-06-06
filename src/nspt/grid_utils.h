#ifndef NSPT_GRID_UTILS_H
#define NSPT_GRID_UTILS_H

/** little convience functions to make Grid types work nicely with other parts
 *  of the code:
 *     * convert grid tensors to util::span
 *     * write grid tensors using fmt library
 */

#include "Grid/Grid.h"
#include "util/span.h"
#include <complex>
#include <fmt/format.h>
#include <sys/stat.h>

/** determine if this is the primary MPI task */
inline bool primaryTask()
{
	return Grid::CartesianCommunicator::RankWorld() == 0;
}

/** check if a file exists */
inline bool fileExists(const std::string &name)
{
	struct stat buffer;
	return stat(name.c_str(), &buffer) == 0;
}

namespace Grid {

util::span<double> asSpan(double &a);
template <typename T> util::span<double> asSpan(iScalar<T> &a);
template <typename T, int N> util::span<double> asSpan(iVector<T, N> &a);
template <typename T, int N> util::span<double> asSpan(iMatrix<T, N> &a);
template <typename T, int N> util::span<double> asSpan(iSeries<T, N> &a);

inline util::span<double> asSpan(double &a) { return {&a, 1}; }
template <typename T> util::span<double> asSpan(iScalar<T> &a)
{
	return asSpan(a());
}

template <typename T, int N> util::span<double> asSpan(iVector<T, N> &a)
{
	auto tmp = asSpan(a(0));
	return {tmp.data(), tmp.size() * N};
}

template <typename T, int N> util::span<double> asSpan(iMatrix<T, N> &a)
{
	auto tmp = asSpan(a(0, 0));
	return {tmp.data(), tmp.size() * N * N};
}

template <typename T, int N> util::span<double> asSpan(iSeries<T, N> &a)
{
	auto tmp = asSpan(a(0));
	return {tmp.data(), tmp.size() * N};
}

util::span<const double> asSpan(const double &a);
template <typename T> util::span<const double> asSpan(const iScalar<T> &a);
template <typename T, int N>
util::span<const double> asSpan(const iVector<T, N> &a);
template <typename T, int N>
util::span<const double> asSpan(const iMatrix<T, N> &a);
template <typename T, int N>
util::span<const double> asSpan(const iSeries<T, N> &a);

inline util::span<const double> asSpan(const double &a) { return {&a, 1}; }
template <typename T> util::span<const double> asSpan(const iScalar<T> &a)
{
	return asSpan(a());
}

template <typename T, int N>
util::span<const double> asSpan(const iVector<T, N> &a)
{
	auto tmp = asSpan(a(0));
	return {tmp.data(), tmp.size() * N};
}

template <typename T, int N>
util::span<const double> asSpan(const iMatrix<T, N> &a)
{
	auto tmp = asSpan(a(0, 0));
	return {tmp.data(), tmp.size() * N * N};
}

template <typename T, int N>
util::span<const double> asSpan(const iSeries<T, N> &a)
{
	auto tmp = asSpan(a(0));
	return {tmp.data(), tmp.size() * N};
}
} // namespace Grid

namespace Grid {
namespace QCD {

/** compute V = exp(aX)U. aliasing is allowed */
inline void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
                   const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp = a * X._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY)U. aliasing is allowed */
inline void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
                   double b, const LatticeGaugeField &Y,
                   const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp = a * X._odata[ss] + b * Y._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY + cZ))U. aliasing is allowed */
inline void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
                   double b, const LatticeGaugeField &Y, double c,
                   const LatticeGaugeField &Z, const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp =
		    a * X._odata[ss] + b * Y._odata[ss] + c * Z._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

} // namespace QCD
} // namespace Grid

namespace fmt {

template <typename T> struct formatter<std::complex<T>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const std::complex<T> &a, FormatContext &ctx)
	{
		auto out = formatter<T>::format(a.real(), ctx);
		out = format_to(out, " + ");
		out = formatter<T>::format(a.imag(), ctx);
		out = format_to(out, "i");
		return out;
	}
};

template <typename T> struct formatter<Grid::iScalar<T>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const Grid::iScalar<T> &a, FormatContext &ctx)
	{
		return formatter<T>::format(a._internal, ctx);
	}
};

template <typename T, int N>
struct formatter<Grid::iVector<T, N>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const Grid::iVector<T, N> &a [[maybe_unused]],
	            FormatContext &ctx)
	{
		auto out = format_to(ctx.out(), "[");
		for (int i = 0; i < N; ++i)
		{
			out = formatter<T>::format(a._internal[i], ctx);
			out = format_to(out, i == N - 1 ? "]" : ", ");
		}
		return out;
	}
};

template <typename T, int N>
struct formatter<Grid::iMatrix<T, N>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const Grid::iMatrix<T, N> &a [[maybe_unused]],
	            FormatContext &ctx)
	{
		auto out = format_to(ctx.out(), "[[");
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				out = formatter<T>::format(a._internal[i][j], ctx);
				if (i == N - 1 && j == N - 1)
					out = format_to(out, "]]");
				else if (j == N - 1)
					out = format_to(out, "], [");
				else
					out = format_to(out, ", ");
			}
		return out;
	}
};

template <typename T, int N>
struct formatter<Grid::iSeries<T, N>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const Grid::iSeries<T, N> &a [[maybe_unused]],
	            FormatContext &ctx)
	{
		auto out = format_to(ctx.out(), "[");
		for (int i = 0; i < N; ++i)
		{
			out = formatter<T>::format(a._internal[i], ctx);
			out = format_to(out, i == N - 1 ? "]" : ", ");
		}
		return out;
	}
};

template <typename T> struct formatter<util::span<T>> : formatter<T>
{
	template <typename FormatContext>
	auto format(const util::span<T> &a, FormatContext &ctx)
	{
		auto out = format_to(ctx.out(), "[");
		for (int i = 0; i < (int)a.size(); ++i)
		{
			out = formatter<T>::format(a[i], ctx);
			out = format_to(out, i == (int)a.size() - 1 ? "]" : ", ");
		}
		return out;
	}
};

} // namespace fmt

#endif
