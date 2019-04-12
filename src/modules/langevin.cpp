#include "modules/langevin.h"

#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/wilson.h"

static void makeNoise(LatticeColourMatrix &out, GridParallelRNG &pRNG,
                      double eps)
{
	gaussian(pRNG, out);
	out *= eps * std::sqrt(0.5); // normalization of SU(3) generators
	out = Ta(out);
}

void MLangevin::run(Environment &env)
{
	// gauge field
	using Field = LatticeColourMatrix;
	std::array<Field, 4> &U = env.store.get<std::array<Field, 4>>(params.field);
	Grid::GridBase *grid = U[0]._grid;

	// temporary storage
	std::array<Field, 4> Uprime{Field(grid), Field(grid), Field(grid),
	                            Field(grid)};
	std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};
	std::array<Field, 4> noise{Field(grid), Field(grid), Field(grid),
	                           Field(grid)};

	// init RNG
	GridParallelRNG pRNG(grid);
	GridSerialRNG sRNG;
	std::vector<int> pseeds({params.seed});
	std::vector<int> sseeds({params.seed + 1});
	pRNG.SeedFixedIntegers(pseeds);
	sRNG.SeedFixedIntegers(sseeds);

	// track some observables during simulation
	std::vector<double> ts, plaq;

	for (int i = 0; i < params.count; ++i)
	{
		// Euler scheme
		if (params.improvement == 0)
		{
			// build force term: F = -eps*D(S) + sqrt(eps/beta) * eta
			for (int mu = 0; mu < 4; ++mu)
			{
				wilsonDeriv(force[mu], U, mu);
				force[mu] *= -params.eps;

				Field drift(grid);
				makeNoise(drift, pRNG,
				          std::sqrt(2.0 * params.eps / params.beta));
				force[mu] += drift;
			}

			// evolve U = exp(F) U
			for (int mu = 0; mu < 4; ++mu)
				U[mu] = expMat(force[mu], 1.0) * U[mu];
		}
		// "BF" scheme
		else if (params.improvement == 1)
		{
			// compute force and noise at U
			for (int mu = 0; mu < 4; ++mu)
			{
				wilsonDeriv(force[mu], U, mu);
				force[mu] *= -params.eps;
				makeNoise(noise[mu], pRNG,
				          std::sqrt(2.0 * params.eps / params.beta));
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
				force[mu] =
				    0.5 * (force[mu] - params.eps * tmp) -
				    (params.eps * params.eps / params.beta * Nc / 6.0) * tmp;
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
		// "Bauer" scheme
		else if (params.improvement == 2)
		{
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
				wilsonDeriv(force[mu], U, mu);
				force[mu] *= params.eps * k1;
				makeNoise(noise[mu], pRNG,
				          std::sqrt(2.0 * params.eps / params.beta));

				force[mu] += k2 * noise[mu];
				Uprime[mu] = expMat(force[mu], 1.0) * U[mu];
			}

			// compute force at U'
			// evolve U = exp(force' + noise) U
			for (int mu = 0; mu < 4; ++mu)
			{
				wilsonDeriv(force[mu], Uprime, mu);
				force[mu] *= params.eps * k4 +
				             params.eps * params.eps / params.beta * Nc * k5;
				force[mu] += k6 * noise[mu];
				U[mu] = expMat(force[mu], -1.0) * U[mu];
			}
		}

		// project to SU(3) in case of rounding errors
		if (params.reunit)
			for (int mu = 0; mu < 4; ++mu)
				ProjectOnGroup(U[mu]);

		// measurements
		double p = avgPlaquette(U);
		ts.push_back((i + 1) * params.eps);
		plaq.push_back(p);

		// some logging
		if ((i + 1) % 10 == 0)
			fmt::print("k = {}/{}, plaq = {}\n", i + 1, params.count, p);
	}

	if (primaryTask() && params.filename != "")
	{
		fmt::print("writing results to '{}'\n", params.filename);

		auto file = DataFile::create(params.filename);
		file.setAttribute("geom", grid->FullDimensions());
		file.setAttribute("count", params.count);
		file.setAttribute("eps", params.eps);
		file.setAttribute("beta", params.beta);
		file.setAttribute("improvement", params.improvement);
		file.setAttribute("reunit", params.reunit);

		file.createData("ts", ts);
		file.createData("plaq", plaq);
	}
}
