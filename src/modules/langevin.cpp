#include "modules/langevin.h"

#include "nspt/grid_utils.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace util;

using GaugeField = QCD::LatticeLorentzColourMatrix;
using GaugeMat = QCD::LatticeColourMatrix;

/** compute V = exp(aX +bY)U. aliasing is allowed */
static void evolve(GaugeField &V, double a, const GaugeField &X, double b,
                   const GaugeField &Y, const GaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		QCD::vLorentzColourMatrix tmp = a * X._odata[ss] + b * Y._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY + cZ))U. aliasing is allowed */
static void evolve(GaugeField &V, double a, const GaugeField &X, double b,
                   const GaugeField &Y, double c, const GaugeField &Z,
                   const GaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		QCD::vLorentzColourMatrix tmp =
		    a * X._odata[ss] + b * Y._odata[ss] + c * Z._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** this creates normal distribution with variance <eta^2>=2 */
static void makeNoise(GaugeField &out, GridParallelRNG &pRNG)
{
	gaussian(pRNG, out);
	out = Ta(out);
}

void MLangevin::run(Environment &env)
{
	// gauge field
	GaugeField &U = env.store.get<GaugeField>(params.field);
	GridBase *grid = U._grid;

	// temporary storage
	GaugeField Uprime(grid);
	GaugeField force(grid);
	GaugeField noise(grid);

	// init RNG
	GridParallelRNG pRNG(grid);
	GridSerialRNG sRNG;
	std::vector<int> pseeds({params.seed});
	pRNG.SeedFixedIntegers(pseeds);

	// track some observables during simulation
	std::vector<double> ts, plaq;

	QCD::WilsonGaugeActionR gaugeAction(1.0);
	double eps = params.eps;
	double beta = params.beta;
	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < params.count; ++i)
	{
		// Euler scheme
		if (params.improvement == 0)
		{
			// build noise and force at U
			makeNoise(noise, pRNG);
			gaugeAction.deriv(U, force);

			// evolve U = exp(F) U
			evolve(U, -eps, force, std::sqrt(eps / beta), noise, U);
		}
		// "BF" scheme
		else if (params.improvement == 1)
		{
			// compute force and noise at U
			gaugeAction.deriv(U, force);
			makeNoise(noise, pRNG);

			// evolve U' = exp(F) U
			evolve(Uprime, -eps, force, std::sqrt(eps / beta), noise, U);

			// compute force at U'
			GaugeField force2(grid);
			gaugeAction.deriv(Uprime, force2);

			// evolve U = exp(F') U
			evolve(U, -0.5 * eps, force + force2, std::sqrt(eps / beta), noise,
			       eps * eps / beta * cA / 6.0, force2, U);
		}
		// "Bauer" scheme
		else if (params.improvement == 2)
		{
			/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
			constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2
			constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2
			constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12

			// compute noise and force at U and evolve U' = exp(F) U
			makeNoise(noise, pRNG);
			gaugeAction.deriv(U, force);
			evolve(Uprime, -eps * k1, force, std::sqrt(eps / beta) * k2, noise,
			       U);

			// compute force at U' and evolve U = exp(F') U
			gaugeAction.deriv(Uprime, force);
			evolve(U, -eps, force, std::sqrt(eps / beta), noise,
			       -k5 * cA * eps * eps / beta, force, U);
		}
		else
			assert(false);

		// project to SU(3) in case of rounding errors
		if (params.reunit)
			for (int mu = 0; mu < 4; ++mu)
				ProjectOnGroup(U);

		// measurements
		double p = QCD::ColourWilsonLoops::avgPlaquette(U);
		ts.push_back((i + 1) * params.eps);
		plaq.push_back(p);

		// some logging
		if ((i + 1) % 10 == 0)
			if (primaryTask())
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
