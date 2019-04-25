#include "nspt/nspt.h"

using namespace Grid;
using Grid::QCD::SpaceTimeGrid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/wilson.h"

using Field = LatticeColourMatrixSeries;
using FieldTerm = LatticeColourMatrix;

/** compute V = exp(aX +bY<<1))U. aliasing is allowed */
static void evolve(Field &V, double a, const Field &X, double b,
                   const FieldTerm &Y, const Field &U)
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
static void evolve(Field &V, double a, const Field &X, double b,
                   const FieldTerm &Y, double c, const Field &Z, const Field &U)
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

LangevinPert::LangevinPert(std::vector<int> latt, int seed)
    : grid(SpaceTimeGrid::makeFourDimGrid(
          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
      pRNG(grid), U{Field(grid), Field(grid), Field(grid), Field(grid)},
      Uprime{Field(grid), Field(grid), Field(grid), Field(grid)}
{
	// start from unit config
	for (int mu = 0; mu < 4; ++mu)
		U[mu] = 1.0;

	std::vector<int> pseeds({seed + 0, seed + 1, seed + 2, seed + 3});
	std::vector<int> sseeds({seed + 4, seed + 5, seed + 6, seed + 7});
	pRNG.SeedFixedIntegers(pseeds);
	sRNG.SeedFixedIntegers(sseeds);
}

/** generate gaussian noise with <eta^2> = 2 */
void LangevinPert::makeNoise(LatticeColourMatrix &out)
{
	// TODO: really check that this produces the correct normalization
	gaussian(pRNG, out);
	out = Ta(out);
	if (flipNoise)
		out = -out;
}

void LangevinPert::evolveStep(double eps)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};
	std::array<FieldTerm, 4> noise{FieldTerm(grid), FieldTerm(grid),
	                               FieldTerm(grid), FieldTerm(grid)};

	// compute noise and force at U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu]);
		wilsonDeriv(force[mu], U, mu);
	}

	// evolve U = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
		evolve(U[mu], -eps, force[mu], std::sqrt(eps), noise[mu], U[mu]);
}

void LangevinPert::evolveStepImproved(double eps)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};
	std::array<FieldTerm, 4> noise{FieldTerm(grid), FieldTerm(grid),
	                               FieldTerm(grid), FieldTerm(grid)};

	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu]);
		wilsonDeriv(force[mu], U, mu);
		evolve(Uprime[mu], -eps, force[mu], std::sqrt(eps), noise[mu], U[mu]);
	}

	// compute force at U' and evolve U = exp(F') U
	for (int mu = 0; mu < 4; ++mu)
	{
		Field tmp(grid);
		wilsonDeriv(tmp, Uprime, mu);
		force[mu] += tmp;
		evolve(U[mu], -eps * 0.5, force[mu], std::sqrt(eps), noise[mu],
		       -eps * eps * Nc / 6.0, tmp, U[mu]);
	}
}

void LangevinPert::evolveStepBauer(double eps)
{
	Field force{Field(grid)};
	std::array<FieldTerm, 4> noise{FieldTerm(grid), FieldTerm(grid),
	                               FieldTerm(grid), FieldTerm(grid)};

	/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
	constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2;
	constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2;
	constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12;

	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu]);
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

void LangevinPert::landauStep(double alpha)
{
	Field R(grid);

	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		// turns out, both of these work fine to suppress the drift in
		// Langevin time. Though Ta() is obviously faster
		Field Amu = logMatFast(U[mu]);
		// Field Amu = Ta(U[mu]);

		R += Amu;
		R -= Cshift(Amu, mu, -1); // NOTE: adj(A) = -A
	}

	/** NOTE: without this projection, the non-anti-hermitian part of A will
	 * grow exponentially in time if no reunitization is done. */
	R = Ta(R);

	R = expMatFast(R, -alpha);

	for (int mu = 0; mu < 4; ++mu)
		U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
}

std::array<LangevinPert::Field, 4> LangevinPert::algebra()
{
	std::array<Field, 4> A{Field(grid), Field(grid), Field(grid), Field(grid)};
	for (int mu = 0; mu < 4; ++mu)
		A[mu] = logMatFast(U[mu]);
	return A;
}

void LangevinPert::zmreg(bool reunit)
{
	Field A{grid};
	for (int mu = 0; mu < 4; ++mu)
	{
		A = logMatFast(U[mu]);
		ColourMatrixSeries avg = sum(A) * (1.0 / A._grid->gSites());
		A = A - avg;

		if (reunit)
			A = Ta(A);

		U[mu] = expMatFast(A, 1.0);
	}
}
