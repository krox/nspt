#include "nspt/langevin.h"

using namespace Grid;
using Grid::QCD::SpaceTimeGrid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/wilson.h"

Langevin::Langevin(std::vector<int> latt, int seed)
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

void Langevin::makeNoise(LatticeColourMatrix &out, double eps)
{
	gaussian(pRNG, out);
	out *= eps * std::sqrt(0.5); // normalization of SU(3) generators
	out = Ta(out);
}

void Langevin::evolveStep(double eps)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};

	// build force term: F = -eps*D(S) + sqrt(eps) * beta^-1/2 * eta

	for (int mu = 0; mu < 4; ++mu)
	{
		// NOTE: this seems to be the performance bottleneck. Not the
		//       "exp(F)" below
		wilsonDeriv(force[mu], U, mu);
		force[mu] *= -eps;

		LatticeColourMatrix drift(grid);
		makeNoise(drift, std::sqrt(2.0 * eps));

		// noise enters the force at order beta^-1/2
		LatticeColourMatrix tmp = peekSeries(force[mu], 1);
		tmp += drift;
		pokeSeries(force[mu], tmp, 1);
	}

	// evolve U = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
		U[mu] = expMatFast(force[mu], 1.0) * U[mu];
}

void Langevin::evolveStepImproved(double eps)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};
	/*std::array<Field, 4> force2{Field(grid), Field(grid), Field(grid),
	                            Field(grid)};*/
	std::array<FieldTerm, 4> noise{FieldTerm(grid), FieldTerm(grid),
	                               FieldTerm(grid), FieldTerm(grid)};

	// compute force and noise at U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force[mu], U, mu);
		force[mu] *= -eps;
		makeNoise(noise[mu], std::sqrt(2.0 * eps));
	}

	// evolve U' = exp(force + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		// noise enters the force at order beta^-1/2
		Field tmp = force[mu];
		LatticeColourMatrix tmp2 = peekSeries(tmp, 1);
		tmp2 += noise[mu];
		pokeSeries(tmp, tmp2, 1);

		Uprime[mu] = expMatFast(tmp, 1.0) * U[mu];
	}

	// build improved force
	for (int mu = 0; mu < 4; ++mu)
	{
		Field tmp(grid);
		wilsonDeriv(tmp, Uprime, mu);
		force[mu] = 0.5 * (force[mu] - eps * tmp) -
		            (eps * eps * Nc / 6.0) * shiftSeries(tmp, 2);
	}

	// evolve U = exp(force' + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		// noise enters the force at order beta^-1/2
		Field tmp = force[mu];
		LatticeColourMatrix tmp2 = peekSeries(tmp, 1);
		tmp2 += noise[mu];
		pokeSeries(tmp, tmp2, 1);

		U[mu] = expMatFast(tmp, 1.0) * U[mu];
	}
}

void Langevin::evolveStepBauer(double eps)
{
	Field force{Field(grid)};
	std::array<FieldTerm, 4> noise{FieldTerm(grid), FieldTerm(grid),
	                               FieldTerm(grid), FieldTerm(grid)};

	/** constant names as in https://arxiv.org/pdf/1303.3279.pdf */
	double k1 = (-3.0 + 2.0 * std::sqrt(2.0)) / 2.0;
	double k2 = (-2.0 + std::sqrt(2.0)) / 2.0;
	// double k3 = 0.0;
	double k4 = 1.0;
	double k5 = (5.0 - 3.0 * std::sqrt(2.0)) / 12.0;
	double k6 = 1.0;

	// compute force and noise at U
	// evolve U' = exp(force + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force, U, mu);
		force *= eps * k1;
		makeNoise(noise[mu], std::sqrt(2.0 * eps));

		// noise enters the force at order beta^-1/2
		LatticeColourMatrix tmp2 = peekSeries(force, 1);
		tmp2 += k2 * noise[mu];
		pokeSeries(force, tmp2, 1);

		Uprime[mu] = expMatFast(force, 1.0) * U[mu];
	}

	// compute force at U'
	// evolve U = exp(force' + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force, Uprime, mu);
		force *= eps * k4;

		// the "C_A" term enters at beta^-1
		force += (k5 * eps * eps * Nc) * shiftSeries(force, 2);

		// noise enters the force at order beta^-1/2
		LatticeColourMatrix tmp2 = peekSeries(force, 1);
		tmp2 += k6 * noise[mu];
		pokeSeries(force, tmp2, 1);

		U[mu] = expMatFast(force, -1.0) * U[mu];
	}
}

void Langevin::landauStep(double alpha)
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
	 * grow exponentially in time */
	R = Ta(R);

	R = expMatFast(R, -alpha);

	for (int mu = 0; mu < 4; ++mu)
		U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
}

std::array<Langevin::Field, 4> Langevin::algebra()
{
	std::array<Field, 4> A{Field(grid), Field(grid), Field(grid), Field(grid)};
	for (int mu = 0; mu < 4; ++mu)
		A[mu] = logMatFast(U[mu]);
	return A;
}

void Langevin::zmreg(bool reunit)
{
	Field A{grid};
	for (int mu = 0; mu < 4; ++mu)
	{
		A = logMatFast(U[mu]);
		ColourMatrixSeries avg = sum(A) * (1.0 / A._grid->gSites());

		// A[i] = A[i] - avg; // This doesn't compile (FIXME)
		std::vector<ColourMatrixSeries> tmp;
		unvectorizeToLexOrdArray(tmp, A);
		for (auto &x : tmp)
			x -= avg;
		vectorizeFromLexOrdArray(tmp, A);

		if (reunit)
			A = Ta(A);

		U[mu] = expMatFast(A, 1.0);
	}
}
