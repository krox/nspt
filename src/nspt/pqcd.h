#ifndef NSPT_PQCD_H
#define NSPT_PQCD_H

#include "Grid/Grid.h"

namespace Grid {

namespace pQCD {

static const int Nd = 4; // space-time dimensions
static const int Ns = 4; // spin-dimensions
static const int Nc = 3; // number of colours
#ifndef NSPT_ORDER
#error Need to specify NSPT_ORDER as macro
#endif
static const int No = NSPT_ORDER; // perturbative expansion

/**
Index order convention: Lorentz x Spin x Series x Colour
**/

/** templated tensor types */

template <typename vtype>
using iSinglet = iScalar<iScalar<iScalar<iScalar<vtype>>>>;
template <typename vtype>
using iSingletSeries = iScalar<iScalar<iSeries<iScalar<vtype>, No>>>;
template <typename vtype>
using iColourMatrix = iScalar<iScalar<iScalar<iMatrix<vtype, Nc>>>>;
template <typename vtype>
using iColourMatrixSeries = iScalar<iScalar<iSeries<iMatrix<vtype, Nc>, No>>>;

/** concrete tensor types (without SIMD) */

typedef iSinglet<Real> TReal;
typedef iSinglet<Complex> TComplex;
typedef iSingletSeries<Real> RealSeries;
typedef iSingletSeries<Complex> ComplexSeries;
typedef iColourMatrix<Complex> ColourMatrix;
typedef iColourMatrixSeries<Complex> ColourMatrixSeries;

/** concrete tensor types (with SIMD) */

typedef iSinglet<vReal> vTReal;
typedef iSinglet<vComplex> vTComplex;
typedef iSingletSeries<vReal> vRealSeries;
typedef iSingletSeries<vComplex> vComplexSeries;
typedef iColourMatrix<vComplex> vColourMatrix;
typedef iColourMatrixSeries<vComplex> vColourMatrixSeries;

/** lattice versions of tensor types (with SIMD) */

typedef Lattice<vTReal> LatticeReal;
typedef Lattice<vTComplex> LatticeComplex;
typedef Lattice<vRealSeries> LatticeRealSeries;
typedef Lattice<vComplexSeries> LatticeComplexSeries;
typedef Lattice<vColourMatrix> LatticeColourMatrix;
typedef Lattice<vColourMatrixSeries> LatticeColourMatrixSeries;

/** Peek and Poke named after physics attributes */

// lorentz
template <class vobj>
auto peekLorentz(const vobj &rhs, int i) -> decltype(PeekIndex<0>(rhs, 0))
{
	return PeekIndex<0>(rhs, i);
}
template <class vobj>
auto peekLorentz(const Lattice<vobj> &rhs, int i)
    -> decltype(PeekIndex<0>(rhs, 0))
{
	return PeekIndex<0>(rhs, i);
}

// spin
template <class vobj>
auto peekSpin(const vobj &rhs, int i) -> decltype(PeekIndex<1>(rhs, 0))
{
	return PeekIndex<1>(rhs, i);
}
template <class vobj>
auto peekSpin(const vobj &rhs, int i, int j)
    -> decltype(PeekIndex<1>(rhs, 0, 0))
{
	return PeekIndex<1>(rhs, i, j);
}
template <class vobj>
auto peekSpin(const Lattice<vobj> &rhs, int i) -> decltype(PeekIndex<1>(rhs, 0))
{
	return PeekIndex<1>(rhs, i);
}
template <class vobj>
auto peekSpin(const Lattice<vobj> &rhs, int i, int j)
    -> decltype(PeekIndex<1>(rhs, 0, 0))
{
	return PeekIndex<1>(rhs, i, j);
}

// series
template <class vobj>
auto peekSeries(const vobj &rhs, int i) -> decltype(peekIndex<2, vobj>(rhs, 0))
{
	return peekIndex<2, vobj>(rhs, i);
}
template <class vobj>
auto peekSeries(const Lattice<vobj> &rhs, int i)
    -> decltype(PeekIndex<2>(rhs, 0))
{
	return PeekIndex<2>(rhs, i);
}

// colour
template <class vobj>
auto peekColour(const vobj &rhs, int i) -> decltype(PeekIndex<3>(rhs, 0))
{
	return PeekIndex<3>(rhs, i);
}
template <class vobj>
auto peekColour(const vobj &rhs, int i, int j)
    -> decltype(PeekIndex<3>(rhs, 0, 0))
{
	return PeekIndex<3>(rhs, i, j);
}
template <class vobj>
auto peekColour(const Lattice<vobj> &rhs, int i)
    -> decltype(PeekIndex<3>(rhs, 0))
{
	return PeekIndex<3>(rhs, i);
}
template <class vobj>
auto peekColour(const Lattice<vobj> &rhs, int i, int j)
    -> decltype(PeekIndex<3>(rhs, 0, 0))
{
	return PeekIndex<3>(rhs, i, j);
}

//////////////////////////////////////////////
// Poke lattice
//////////////////////////////////////////////
template <class vobj>
void pokeColour(Lattice<vobj> &lhs,
                const Lattice<decltype(peekIndex<3>(lhs._odata[0], 0))> &rhs,
                int i)
{
	PokeIndex<3>(lhs, rhs, i);
}
template <class vobj>
void pokeColour(Lattice<vobj> &lhs,
                const Lattice<decltype(peekIndex<3>(lhs._odata[0], 0, 0))> &rhs,
                int i, int j)
{
	PokeIndex<3>(lhs, rhs, i, j);
}
template <class vobj>
void pokeSpin(Lattice<vobj> &lhs,
              const Lattice<decltype(peekIndex<1>(lhs._odata[0], 0))> &rhs,
              int i)
{
	PokeIndex<1>(lhs, rhs, i);
}
template <class vobj>
void pokeSpin(Lattice<vobj> &lhs,
              const Lattice<decltype(peekIndex<1>(lhs._odata[0], 0, 0))> &rhs,
              int i, int j)
{
	PokeIndex<1>(lhs, rhs, i, j);
}
template <class vobj>
void pokeLorentz(Lattice<vobj> &lhs,
                 const Lattice<decltype(peekIndex<0>(lhs._odata[0], 0))> &rhs,
                 int i)
{
	PokeIndex<0>(lhs, rhs, i);
}

template <class vobj>
void pokeSeries(Lattice<vobj> &lhs,
                const Lattice<decltype(peekIndex<2>(lhs._odata[0], 0))> &rhs,
                int i)
{
	PokeIndex<2>(lhs, rhs, i);
}

//////////////////////////////////////////////
// Poke scalars
//////////////////////////////////////////////
template <class vobj>
void pokeSpin(vobj &lhs, const decltype(peekIndex<1>(lhs, 0)) &rhs, int i)
{
	pokeIndex<1>(lhs, rhs, i);
}
template <class vobj>
void pokeSpin(vobj &lhs, const decltype(peekIndex<1>(lhs, 0, 0)) &rhs, int i,
              int j)
{
	pokeIndex<1>(lhs, rhs, i, j);
}

template <class vobj>
void pokeColour(vobj &lhs, const decltype(peekIndex<3>(lhs, 0)) &rhs, int i)
{
	pokeIndex<3>(lhs, rhs, i);
}
template <class vobj>
void pokeColour(vobj &lhs, const decltype(peekIndex<3>(lhs, 0, 0)) &rhs, int i,
                int j)
{
	pokeIndex<3>(lhs, rhs, i, j);
}

template <class vobj>
void pokeLorentz(vobj &lhs, const decltype(peekIndex<0>(lhs, 0)) &rhs, int i)
{
	pokeIndex<0>(lhs, rhs, i);
}

template <class vobj>
void pokeSeries(vobj &lhs, const decltype(peekIndex<2>(lhs, 0)) &rhs, int i)
{
	pokeIndex<2>(lhs, rhs, i);
}

// transpose array and scalar

template <class vobj>
inline Lattice<vobj> transposeSpin(const Lattice<vobj> &lhs)
{
	return transposeIndex<1>(lhs);
}

template <class vobj>
inline Lattice<vobj> transposeColour(const Lattice<vobj> &lhs)
{
	return transposeIndex<3>(lhs);
}

template <class vobj> inline vobj transposeSpin(const vobj &lhs)
{
	return transposeIndex<1>(lhs);
}

template <class vobj> inline vobj transposeColour(const vobj &lhs)
{
	return transposeIndex<3>(lhs);
}

// Trace lattice and non-lattice

template <class vobj>
inline auto traceSpin(const Lattice<vobj> &lhs)
    -> Lattice<decltype(traceIndex<1>(lhs._odata[0]))>
{
	return traceIndex<1>(lhs);
}
template <class vobj>
inline auto traceColour(const Lattice<vobj> &lhs)
    -> Lattice<decltype(traceIndex<3>(lhs._odata[0]))>
{
	return traceIndex<3>(lhs);
}
template <class vobj>
inline auto traceSpin(const vobj &lhs) -> Lattice<decltype(traceIndex<1>(lhs))>
{
	return traceIndex<1>(lhs);
}
template <class vobj>
inline auto traceColour(const vobj &lhs)
    -> Lattice<decltype(traceIndex<3>(lhs))>
{
	return traceIndex<3>(lhs);
}

} // namespace pQCD

} // namespace Grid

#endif
