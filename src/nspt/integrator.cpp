#include "nspt/integrator.h"

using namespace Grid;
using namespace pQCD;

#include "nspt/wilson.h"

/** generate gaussian noise with <eta^2> = 2 */
static void makeNoise(LatticeColourMatrix &out, GridParallelRNG &pRNG)
{
	// TODO: really check that this produces the correct normalization
	gaussian(pRNG, out);
	out = Ta(out);
}

/** compute V = exp(aX +bY<<1))U. aliasing is allowed */
static void evolve(LatticeColourMatrixSeries &V, double a,
                   const LatticeColourMatrixSeries &X, double b,
                   const LatticeColourMatrix &Y,
                   const LatticeColourMatrixSeries &U)
{
	assert(No >= 2);
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vColourMatrixSeries tmp = a * X._odata[ss];
		tmp()()(1) += b * Y._odata[ss]()()();
		V._odata[ss] = ExponentiateFast(tmp, 1.0) * U._odata[ss];
	}
}

/** compute V = exp(aX +bY<<2 + cZ<<4))U. aliasing is allowed */
static void evolve(LatticeColourMatrixSeries &V, double a,
                   const LatticeColourMatrixSeries &X, double b,
                   const LatticeColourMatrix &Y, double c,
                   const LatticeColourMatrixSeries &Z,
                   const LatticeColourMatrixSeries &U)
{
	assert(No >= 2);
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vColourMatrixSeries tmp = a * X._odata[ss];
		tmp()()(1) += b * Y._odata[ss]()()();
		for (int i = 2; i < No; ++i)
			tmp()()(i) += c * Z._odata[ss]()()(i - 2);
		V._odata[ss] = ExponentiateFast(tmp, 1.0) * U._odata[ss];
	}
}

EulerIntegrator::EulerIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{grid}, force{Field(grid), Field(grid), Field(grid),
                                     Field(grid)}
{
	pRNG.SeedFixedIntegers({seed});
}

void EulerIntegrator::step(std::array<Field, 4> &U, double eps)
{
	// compute force at U
	for (int mu = 0; mu < 4; ++mu)
		wilsonDeriv(force[mu], U, mu);

	// evolve U = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise, pRNG);
		evolve(U[mu], -eps, force[mu], std::sqrt(eps), noise, U[mu]);
	}
}

BFIntegrator::BFIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{Term(grid), Term(grid), Term(grid), Term(grid)},
      force{Field(grid), Field(grid), Field(grid), Field(grid)},
      Uprime{Field(grid), Field(grid), Field(grid), Field(grid)}, tmp{grid}
{
	pRNG.SeedFixedIntegers({seed});
}

void BFIntegrator::step(std::array<Field, 4> &U, double eps)
{
	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu], pRNG);
		wilsonDeriv(force[mu], U, mu);
		evolve(Uprime[mu], -eps, force[mu], std::sqrt(eps), noise[mu], U[mu]);
	}

	// compute force at U' and evolve U = exp(F') U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(tmp, Uprime, mu);
		force[mu] += tmp;
		evolve(U[mu], -eps * 0.5, force[mu], std::sqrt(eps), noise[mu],
		       -eps * eps * Nc / 6.0, tmp, U[mu]);
	}
}

BauerIntegrator::BauerIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{Term(grid), Term(grid), Term(grid), Term(grid)},
      force{grid}, Uprime{Field(grid), Field(grid), Field(grid), Field(grid)}
{
	pRNG.SeedFixedIntegers({seed});
}

void BauerIntegrator::step(std::array<Field, 4> &U, double eps)
{
	/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
	constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2;
	constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2;
	constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12;

	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu], pRNG);
		wilsonDeriv(force, U, mu);
		evolve(Uprime[mu], -eps * k1, force, std::sqrt(eps) * k2, noise[mu],
		       U[mu]);
	}

	// compute force at U' and evolve U = exp(F') U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force, Uprime, mu);
		evolve(U[mu], -eps, force, std::sqrt(eps), noise[mu],
		       -k5 * Nc * eps * eps, force, U[mu]);
	}
}
