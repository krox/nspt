#include "modules/sweep_beta.h"

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

void MSweepBeta::run(Environment &env)
{
	// gauge field
	GridCartesian *grid = env.getGrid(params.grid);
	GridRedBlackCartesian *gridRB = env.getGridRB(params.grid);
	GaugeField U(grid);
	U = 1.0;

	// init RNG
	GridParallelRNG pRNG(grid);
	std::vector<int> pseeds({params.seed});
	pRNG.SeedFixedIntegers(pseeds);
	GridSerialRNG sRNG;
	sRNG.SeedFixedIntegers({params.seed + 1});

	// track some observables during simulation
	std::vector<double> bs, plaq, loop;

	for (int k = 0; k < params.beta_steps; ++k)
	{
		// create action (NOTE: dont do beta=0. bad for rescaling)
		actionParams["beta"] = params.beta_max / params.beta_steps * (k + 1);
		CompositeAction<GaugeField> action(actionParams, grid, gridRB);
		std::cout << action.LogParameters() << std::endl;
		auto integrator = makeQCDIntegrator(action, integratorParams);

		// first half of sweeps for thermalization
		integrator->run(U, sRNG, pRNG, params.sweeps / 2);

		// second half with measurements
		integrator->resetStats();
		integrator->run(U, sRNG, pRNG, params.sweeps / 2);
		double p = mean(integrator->plaq_history);
		double l = mean(integrator->loop_history);

		// log results
		bs.push_back(action.beta);
		plaq.push_back(p);
		loop.push_back(l);
		if (primaryTask())
			fmt::print("beta = {}, plaq = {}, loop = {}\n", action.beta, p, l);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(bs, plaq, "plaq").plotData(bs, loop, "loop");
	}
}
