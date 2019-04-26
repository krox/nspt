#include "Grid/Grid.h"

#include "util/json.hpp"

#include "util/CLI11.hpp"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/stopwatch.h"
#include "util/vector2d.h"
#include <fmt/format.h>

#include "nspt/grid_utils.h"
#include "nspt/integrator.h"
#include "nspt/nspt.h"
#include "nspt/pqcd.h"

using namespace Grid;
using Grid::QCD::SpaceTimeGrid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/algebra_observables.h"
#include "nspt/wilson.h"

using namespace nlohmann;

int main(int argc, char **argv)
{
	using Field = LatticeColourMatrixSeries;

	// parameters
	int count = 400;
	int discard = 0;
	double eps = 0.05;

	int improvement = 1;
	int gaugefix = 1;
	int zmreg = 1;
	int reunit = 1;

	int doPlot = 0;
	int seed = -1;
	int verbosity = 1;
	bool plotSeparate = false;

	std::string filename;

	// read command line
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--count", count, "number of configs to generate");
	app.add_option("--discard", discard, "thermalization");
	app.add_option("--eps", eps, "stepsize for integration");
	app.add_option("--improvement", improvement, "use improved integrator");
	app.add_option("--gaugefix", gaugefix, "do stochastic gauge fixing");
	app.add_option("--zmreg", zmreg, "explicitly remove zero modes");
	app.add_option("--reunit", reunit, "explicitly project onto group/algebra");
	app.add_option("--threads", dummy);
	app.add_option("--mpi", dummy);
	app.add_option("--plot", doPlot, "show a plot (requires Gnuplot)");
	app.add_flag("--plot-separate", plotSeparate, "plot orders separately");
	app.add_option("--filename", filename, "output file (json format)");
	app.add_option("--seed", seed, "seed for rng (default = unpredictable)");
	app.add_option("--verbosity", verbosity, "verbosity (default = 1)");
	CLI11_PARSE(app, argc, argv);

	if (filename != "" && fileExists(filename))
	{
		if (primaryTask())
			fmt::print("{} already exists. skipping.\n", filename);
		return 0;
	}

	if (filename != "" && filename.find(".h5") == std::string::npos)
	{
		if (primaryTask())
			fmt::print("ERROR: unrecognized file ending: {}\n", filename);
		return -1;
	}

	Grid_init(&argc, &argv);
	std::vector<int> geom = GridDefaultLatt();
	if (seed == -1)
		seed = std::random_device()();

	if (primaryTask())
		fmt::print("L = {}, eps = {}, maxt = {}\n", geom[0], eps,
		           (count + discard) * eps);

	// initialize the gauge field
	GridBase *grid = SpaceTimeGrid::makeFourDimGrid(
	    geom, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
	grid->show_decomposition();
	std::array<Field, 4> U = {Field(grid), Field(grid), Field(grid),
	                          Field(grid)};
	for (int mu = 0; mu < 4; ++mu)
		U[mu] = 1.0;

	// temporary
	std::array<Field, 4> A = {Field(grid), Field(grid), Field(grid),
	                          Field(grid)};

	// some observables to track
	std::vector<double> ts;
	vector2d<double> plaq;
	AlgebraObservables algObs;

	// performance measure
	Stopwatch swEvolve, swMeasure, swLandau, swZmreg;

	// evolve it some time
	std::unique_ptr<Integrator> integrator;
	if (improvement == 0)
		integrator = std::make_unique<EulerIntegrator>(grid, seed);
	else if (improvement == 1)
		integrator = std::make_unique<BFIntegrator>(grid, seed);
	else if (improvement == 2)
		integrator = std::make_unique<BauerIntegrator>(grid, seed);
	else
		assert(false);

	for (int k = -discard; k < count; ++k)
	{
		// step 1: langevin evolution
		swEvolve.start();
		integrator->step(U, eps);
		swEvolve.stop();

		// step 2: stochastic gauge-fixing and zero-mode regularization
		// NOTE: the precise amount of gauge-fixing is somewhat arbitrary, but
		//       don't scale it with epsilon.

		swLandau.start();
		for (int i = 0; i < gaugefix; ++i)
			landauStep(U, 0.1);
		swLandau.stop();

		if (zmreg)
		{
			swZmreg.start();
			removeZero(U, reunit);
			swZmreg.stop();
		}

		// step 3: measurements

		swMeasure.start();

		double t = (k + discard + 1) * eps; // Langevin time
		RealSeries p = avgPlaquette(U);     // plaquette

		if (verbosity >= 2 || (verbosity >= 1 && k % 10 == 0))
			if (primaryTask())
				fmt::print("k = {}/{} t = {}, plaq = {:.5}\n", k, count, t, p);

		if (k >= 0)
		{
			ts.push_back(t);
			plaq.push_back(asSpan(p));

			// trace/hermiticity/etc of algebra
			for (int mu = 0; mu < 4; ++mu)
				A[mu] = logMatFast(U[mu]);
			algObs.measure(A);
		}

		swMeasure.stop();
	}

	if (primaryTask())
	{
		if (filename != "")
		{
			fmt::print("writing results to '{}'\n", filename);

			auto file = DataFile::create(filename);
			file.setAttribute("order", No);
			file.setAttribute("geom", geom);
			file.setAttribute("count", count);
			file.setAttribute("discard", discard);
			file.setAttribute("eps", eps);
			file.setAttribute("improvement", improvement);
			file.setAttribute("gaugefix", gaugefix);
			file.setAttribute("zmreg", zmreg);
			file.setAttribute("reunit", reunit);

			file.createData("ts", ts);
			file.createData("plaq", plaq);
			file.createData("traceA", algObs.traceA);
			file.createData("hermA", algObs.hermA);
			file.createData("normA", algObs.normA);
			file.createData("gaugeCond", algObs.gaugeCond);
			file.createData("avgAx", algObs.avgAx);
			file.createData("avgAy", algObs.avgAy);
			file.createData("avgAz", algObs.avgAz);
			file.createData("avgAt", algObs.avgAt);
		}

		fmt::print("time for Langevin evolution: {}\n", swEvolve.secs());
		fmt::print("time for Landau gaugefix: {}\n", swLandau.secs());
		fmt::print("time for ZM regularization: {}\n", swZmreg.secs());
		fmt::print("time for measurments: {}\n", swMeasure.secs());

		if (doPlot >= 1)
		{
			if (plotSeparate)
				for (int i = 2; i < No; i += 2)
					Gnuplot().style("lines").plotData(
					    ts, plaq.col(i), fmt::format("plaq[{}]", i));
			else
				Gnuplot().style("lines").plotData(ts, plaq, "plaq");
		}
		if (doPlot >= 2)
			algObs.plot(ts);
	}

	Grid_finalize();
}
