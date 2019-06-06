#include "modules/hmc.h"

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

void MHMC::run(Environment &env)
{
	// gauge field
	GaugeField &U = env.store.get<GaugeField>(params.field);
	GridCartesian *grid = env.getGrid(U._grid->FullDimensions());
	GridRedBlackCartesian *gridRB = env.getGridRB(U._grid->FullDimensions());

	// temporary storage
	GaugeField mom(grid);
	GaugeField force(grid);

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

	for (int i = 0; i < params.count; ++i)
	{
		// numerical integration of the hmc process
		for (int iter = 0; iter < params.sweeps; ++iter)
		{
			// new pseudo-fermions and momenta
			action.refresh(U, pRNG);
			gaussian(pRNG, mom);
			mom = std::sqrt(0.5) * Ta(mom); // TODO: right scale here?

			// leap-frog integration
			QCD::evolve(U, 0.5 * params.eps, mom, U);
			action.deriv(U, force);
			force *= -params.eps;
			mom += force;
			QCD::evolve(U, 0.5 * params.eps, mom, U);
		}

		// project to SU(3) to fix rounding errors
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
