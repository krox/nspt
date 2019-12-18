#include "modules/markov.h"

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

void MMarkov::run(Environment &env)
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
	std::vector<double> plaq, loop;

	// create the action
	CompositeAction<GaugeField> action(actionParams, grid, gridRB);
	std::cout << action.LogParameters() << std::endl;

	// create the integrator
	std::unique_ptr<QCDIntegrator> integrator =
	    makeQCDIntegrator(action, integratorParams);

	for (int i = 0; i < params.count; ++i)
	{
		// numerical integration of the langevin process
		integrator->run(U, sRNG, pRNG, params.sweeps);

		// measurements
		double p = QCD::ColourWilsonLoops::avgPlaquette(U);
		double l = real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
		plaq.push_back(p);
		loop.push_back(l);

		// some logging
		if (primaryTask())
			fmt::print("k = {}/{}, plaq = {}, loop = {}\n", i + 1, params.count,
			           p, l);

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

		// integrator params
		file.setAttribute("method",
		                  integratorParams.value("method", "unknown"));
		file.setAttribute("epsilon",
		                  integratorParams.value("epsilon", 0.0 / 0.0));

		// action params
		file.setAttribute("gauge_action", action.gauge_action);
		file.setAttribute("beta", action.beta);
		file.setAttribute("fermion_action", action.fermion_action);
		file.setAttribute("csw", action.csw);
		file.setAttribute("kappa_light", action.kappa_light);

		// measurments
		file.createData("plaq", plaq);
		file.createData("loop", loop);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(plaq, "plaq"); //.plotData(loop, "loop");
	}
}
