#include "modules/langevin.h"

#include "nspt/action.h"
#include "nspt/grid_utils.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace util;

using GaugeField = QCD::LatticeLorentzColourMatrix;
using GaugeMat = QCD::LatticeColourMatrix;
using FermionField = QCD::LatticeSpinColourVector;

/** this creates normal distribution with variance <eta^2>=2 */
static void makeNoise(GaugeField &out, GridParallelRNG &pRNG)
{
	gaussian(pRNG, out);
	out = Ta(out);
	auto n = norm2(out) / out._grid->gSites();
	if (primaryTask())
		fmt::print("noise force = {}\n", n);
}

/** NOTE: this does basic non-rescaled Langevin evolution.
 * For nice correlations, use eps=eps/beta so that the effective drift is
 * invariant of beta */
static void integrateLangevin(GaugeField &U, QCD::Action<GaugeField> &action,
                              GridParallelRNG &pRNG, double eps, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField noise(grid);

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		QCD::evolve(U, -eps, force, std::sqrt(eps), noise, U);
	}
}

static void integrateLangevinBF(GaugeField &U, QCD::Action<GaugeField> &action,
                                GridParallelRNG &pRNG, double eps, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField force2(grid);
	GaugeField noise(grid);
	GaugeField Uprime(grid);

	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		// compute force and noise at U
		action.deriv(U, force);
		force = Ta(force);
		makeNoise(noise, pRNG);

		// evolve U' = exp(F) U
		QCD::evolve(Uprime, -eps, force, std::sqrt(eps), noise, U);

		// compute force at U'
		action.deriv(Uprime, force2);
		force2 = Ta(force2);

		// evolve U = exp(F') U
		QCD::evolve(U, -0.5 * eps, force + force2, std::sqrt(eps), noise,
		            eps * eps * cA / 6.0, force2, U);
	}
}

static void integrateLangevinBauer(GaugeField &U,
                                   QCD::Action<GaugeField> &action,
                                   GridParallelRNG &pRNG, double eps,
                                   int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField noise(grid);
	GaugeField Uprime(grid);

	/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
	constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2
	constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2
	constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12

	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		// compute noise and force at U and evolve U' = exp(F) U
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		QCD::evolve(Uprime, -eps * k1, force, std::sqrt(eps) * k2, noise, U);

		// compute force at U' and evolve U = exp(F') U
		action.deriv(Uprime, force);
		force = Ta(force);
		QCD::evolve(U, -eps - k5 * cA * eps * eps, force, std::sqrt(eps), noise,
		            U);
	}
}

void MLangevin::run(Environment &env)
{
	// gauge field
	GaugeField &U = env.store.get<GaugeField>(params.field);
	GridCartesian *grid = env.getGrid(U._grid->FullDimensions());
	GridRedBlackCartesian *gridRB = env.getGridRB(U._grid->FullDimensions());

	// temporary storage
	GaugeField Uprime(grid);
	GaugeField force(grid);
	GaugeField noise(grid);

	// init RNG
	GridParallelRNG pRNG(grid);
	GridSerialRNG sRNG;
	std::vector<int> pseeds({params.seed});
	pRNG.SeedFixedIntegers(pseeds);
	if (params.rng != "")
	{
		FieldMetaData header;
		QCD::NerscIO::readRNGState(sRNG, pRNG, header, params.rng);
	}

	// track some observables during simulation
	std::vector<double> ts, plaq;

	CompositeAction<GaugeField> action(actionParams, grid, gridRB);
	std::cout << action.LogParameters() << std::endl;

	// rescale step size
	double delta = params.eps / action.beta;

	for (int i = 0; i < params.count; ++i)
	{
		// numerical integration of the langevin process
		if (params.improvement == 0)
			integrateLangevin(U, action, pRNG, delta, params.sweeps);
		else if (params.improvement == 1)
			integrateLangevinBF(U, action, pRNG, delta, params.sweeps);
		else if (params.improvement == 2)
			integrateLangevinBauer(U, action, pRNG, delta, params.sweeps);
		else
			assert(false);

		// project to SU(3) in case of rounding errors
		if (params.reunit)
			ProjectOnGroup(U);

		// measurements
		double p = QCD::ColourWilsonLoops::avgPlaquette(U);
		ts.push_back((i + 1) * params.eps);
		plaq.push_back(p);

		// some logging
		if (primaryTask())
			fmt::print("k = {}/{}, plaq = {}\n", i + 1, params.count, p);

		// write config to file
		if (params.path != "")
		{
			std::string filename =
			    fmt::format("{}/{}{}.cnf", params.path, params.prefix, i + 1);
			if (primaryTask())
				fmt::print("writing config to {}.nersc\n", filename);
			QCD::NerscIO::writeConfiguration(U, filename, 0, 0);
		}
		// write config to file
		if (params.path != "")
		{
			std::string filename =
			    fmt::format("{}/{}{}.rng", params.path, params.prefix, i + 1);
			if (primaryTask())
				fmt::print("writing rng to {}\n", filename);
			QCD::NerscIO::writeRNGState(sRNG, pRNG, filename);
		}
	}

	if (primaryTask() && params.filename != "")
	{
		fmt::print("writing results to '{}'\n", params.filename);

		auto file = DataFile::create(params.filename);
		file.setAttribute("geom", grid->FullDimensions());
		file.setAttribute("count", params.count);

		file.setAttribute("sweeps", params.sweeps);
		file.setAttribute("eps", params.eps);
		file.setAttribute("improvement", params.improvement);

		file.setAttribute("gauge_action", action.gauge_action);
		file.setAttribute("beta", action.beta);

		file.setAttribute("fermion_action", action.fermion_action);
		file.setAttribute("csw", action.csw);
		file.setAttribute("kappa_light", action.kappa_light);

		file.setAttribute("reunit", params.reunit);

		file.createData("ts", ts);
		file.createData("plaq", plaq);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(plaq, "plaq");
	}
}
