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

void Langevin::makeNoise(Field &out, double eps)
{
	gaussian(pRNG, out);
	out *= eps * std::sqrt(0.5); // normalization of SU(3) generators
	out = Ta(out);
	if (flipNoise)
		out = -out;
}

void Langevin::evolveStep(double eps, double beta)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};

	// build force term: F = -eps*beta*D(S) + sqrt(eps) * eta
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force[mu], U, mu);
		force[mu] *= -eps * beta;

		Field drift(grid);
		makeNoise(drift, std::sqrt(2.0 * eps));
		force[mu] += drift;
	}

	// evolve U = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
		U[mu] = expMat(force[mu], 1.0) * U[mu];
}

void Langevin::evolveStepImproved(double eps, double beta)
{
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};
	std::array<Field, 4> noise{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};

	// compute force and noise at U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force[mu], U, mu);
		force[mu] *= -eps * beta;
		makeNoise(noise[mu], std::sqrt(2.0 * eps));
	}

	// evolve U' = exp(force + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		// noise enters the force at order beta^-1/2
		Field tmp = force[mu];
		tmp += noise[mu];
		Uprime[mu] = expMat(tmp, 1.0) * U[mu];
	}

	// build improved force
	for (int mu = 0; mu < 4; ++mu)
	{
		Field tmp(grid);
		wilsonDeriv(tmp, Uprime, mu);
		force[mu] = 0.5 * (force[mu] - (eps * beta) * tmp) -
		            (eps * eps * beta * Nc / 6.0) * tmp;
	}

	// evolve U = exp(force' + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		// noise enters the force at order beta^-1/2
		Field tmp = force[mu];
		tmp += noise[mu];
		U[mu] = expMat(tmp, 1.0) * U[mu];
	}
}

void Langevin::evolveStepBauer(double eps, double beta)
{
	Field force{Field(grid)};
	std::array<Field, 4> noise{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};

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
		force *= (eps * beta) * k1;
		makeNoise(noise[mu], std::sqrt(2.0 * eps));

		force += k2 * noise[mu];
		Uprime[mu] = expMat(force, 1.0) * U[mu];
	}

	// compute force at U'
	// evolve U = exp(force' + noise) U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force, Uprime, mu);
		force *= beta * eps * k4 + beta * eps * eps * Nc * k5;
		force += k6 * noise[mu];
		U[mu] = expMat(force, -1.0) * U[mu];
	}
}
