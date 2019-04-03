#include "Grid/Grid.h"

#include "util/json.hpp"

#include "util/CLI11.hpp"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/stopwatch.h"
#include "util/vector2d.h"
#include <experimental/filesystem>
#include <fmt/format.h>

#include "nspt/grid_utils.h"
#include "nspt/langevin.h"
#include "nspt/pqcd.h"

using namespace Grid;
using Grid::QCD::SpaceTimeGrid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/wilson.h"

using namespace nlohmann;

int main(int argc, char **argv)
{
	// parameters
	int count = 400;
	int discard = 0;
	double eps = 0.05;
	double beta = 6.0;

	int improvement = 1;
	int reunit = 1;

	int doPlot = 0;
	int seed = -1;
	int verbosity = 1;

	std::string filename;
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--count", count, "number of configs to generate");
	app.add_option("--discard", discard, "thermalization");
	app.add_option("--eps", eps, "stepsize for integration");
	app.add_option("--beta", beta, "(inverse) coupling");
	app.add_option("--improvement", improvement, "use improved integrator");
	app.add_option("--reunit", reunit, "explicitly project onto group");
	app.add_option("--threads", dummy);
	app.add_option("--plot", doPlot, "show a plot (requires Gnuplot)");
	app.add_option("--filename", filename, "output file (json format)");
	app.add_option("--seed", seed, "seed for rng (default = unpredictable)");
	app.add_option("--verbosity", verbosity, "verbosity (default = 1)");
	CLI11_PARSE(app, argc, argv);

	if (filename != "" && std::experimental::filesystem::exists(filename))
	{
		fmt::print("{} already exists. skipping.\n", filename);
		return 0;
	}

	if (filename != "" && filename.find(".h5") == std::string::npos)
	{
		fmt::print("ERROR: unrecognized file ending: {}\n", filename);
		return -1;
	}

	Grid_init(&argc, &argv);
	std::vector<int> geom = GridDefaultLatt();

	fmt::print("L = {}, beta = {} eps = {}, maxt = {}\n", geom[0], beta, eps,
	           (count + discard) * eps);

	// data
	if (seed == -1)
		seed = std::random_device()();
	auto lang = Langevin(geom, seed);
	std::vector<double> ts;
	std::vector<double> plaq;

	// performance measure
	Stopwatch swEvolve, swMeasure;

	// evolve it some time
	for (int k = -discard; k < count; ++k)
	{
		// step 1: langevin evolution
		swEvolve.start();
		if (improvement == 0)
			lang.evolveStep(eps, beta);
		else if (improvement == 1)
			lang.evolveStepImproved(eps, beta);
		else if (improvement == 2)
			lang.evolveStepBauer(eps, beta);
		else
			assert(false);
		swEvolve.stop();

		if (reunit)
			for (int mu = 0; mu < 4; ++mu)
				ProjectOnGroup(lang.U[mu]);

		// step 2: measurements
		swMeasure.start();

		double t = (k + discard + 1) * eps; // Langevin time
		double p = avgPlaquette(lang.U);    // plaquette

		if (verbosity >= 2 || (verbosity >= 1 && k % 10 == 0))
			fmt::print("k = {}/{} t = {}, plaq = {:.5}\n", k, count, t, p);

		if (k >= 0)
		{
			ts.push_back(t);
			plaq.push_back(p);
		}

		swMeasure.stop();
	}

	if (filename != "")
	{
		fmt::print("writing results to '{}'\n", filename);

		auto file = DataFile::create(filename);
		file.setAttribute("geom", geom);
		file.setAttribute("count", count);
		file.setAttribute("discard", discard);
		file.setAttribute("eps", eps);
		file.setAttribute("beta", beta);
		file.setAttribute("improvement", improvement);
		file.setAttribute("reunit", reunit);

		file.createData("ts", ts);
		file.createData("plaq", plaq);
	}

	fmt::print("time for Langevin evolution: {}\n", swEvolve.secs());
	fmt::print("time for measurments: {}\n", swMeasure.secs());

	if (doPlot >= 1)
		Gnuplot().style("lines").plotData(ts, plaq, "plaq");

	Grid_finalize();
}
