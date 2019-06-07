#include "modules/langevin.h"

#include "nspt/action.h"
#include "nspt/grid_utils.h"
#include "qcd/evolution.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace util;

using GaugeField = QCD::LatticeLorentzColourMatrix;
using GaugeMat = QCD::LatticeColourMatrix;
using FermionField = QCD::LatticeSpinColourVector;

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
		double _;
		if (params.improvement == 0)
			QCD::integrateLangevin(U, action, sRNG, pRNG, delta, params.sweeps,
			                       _, _);
		else if (params.improvement == 1)
			QCD::integrateLangevinBF(U, action, sRNG, pRNG, delta,
			                         params.sweeps, _, _);
		else if (params.improvement == 2)
			QCD::integrateLangevinBauer(U, action, sRNG, pRNG, delta,
			                            params.sweeps, _, _);
		else
			assert(false);

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

		file.createData("ts", ts);
		file.createData("plaq", plaq);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(plaq, "plaq");
	}
}
