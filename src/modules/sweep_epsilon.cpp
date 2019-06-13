#include "modules/sweep_epsilon.h"

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

void MSweepEpsilon::run(Environment &env)
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

	CompositeAction<GaugeField> action(actionParams, grid, gridRB);
	std::cout << action.LogParameters() << std::endl;

	// track some observables during simulation
	std::vector<double> es, plaq1, plaq2, loop1, loop2;

	for (int k = 0; k < params.epsilon_steps; ++k)
	{
		// create action (NOTE: dont do beta=0. bad for rescaling)
		double eps =
		    params.epsilon_min + (params.epsilon_max - params.epsilon_min) * k /
		                             (params.epsilon_steps - 1);
		integratorParams["epsilon"] = eps;

		auto integrator = makeQCDIntegrator(action, integratorParams);

		// first third for thermalization
		integrator->run(U, sRNG, pRNG, params.sweeps / 3);

		// second third with measruements
		integrator->resetStats();
		integrator->run(U, sRNG, pRNG, params.sweeps / 3);
		double p1 = mean(integrator->plaq_history);
		double l1 = mean(integrator->loop_history);

		// third third with measruements
		integrator->resetStats();
		integrator->run(U, sRNG, pRNG, params.sweeps / 3);
		double p2 = mean(integrator->plaq_history);
		double l2 = mean(integrator->loop_history);

		// log results
		es.push_back(eps);
		plaq1.push_back(p1);
		loop1.push_back(l1);
		plaq2.push_back(p2);
		loop2.push_back(l2);
		if (primaryTask())
			fmt::print("eps = {}, plaq = {}, loop = {}, acc = {:.3f}\n", eps,
			           p2, l2, integrator->acceptance());
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(es, plaq1, "plaq1").plotData(es, plaq2, "plaq2");
		Gnuplot().plotData(es, loop1, "loop1").plotData(es, loop2, "loop2");
	}
}
