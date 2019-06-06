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

	// track some observables during simulation
	std::vector<double> bs, plaq;

	for (int k = 0; k < params.beta_steps; ++k)
	{
		// create action (NOTE: dont do beta=0. bad for rescaling)
		actionParams["beta"] = params.beta_max / params.beta_steps * (k + 1);
		CompositeAction<GaugeField> action(actionParams, grid, gridRB);
		std::cout << action.LogParameters() << std::endl;

		// rescale step size
		double delta = params.eps / action.beta;

		double p = 0.0;

		for (int i = 0; i < params.sweeps; ++i)
		{
			if (params.method == "LangevinEuler")
				QCD::integrateLangevin(U, action, pRNG, delta, 1);
			else if (params.method == "LangevinBF")
				QCD::integrateLangevinBF(U, action, pRNG, delta, 1);
			else if (params.method == "LangevinBauer")
				QCD::integrateLangevinBauer(U, action, pRNG, delta, 1);
			else if (params.method == "HMC")
				QCD::integrateHMC(U, action, pRNG, delta, 1);
			else
				assert(false);

			ProjectOnGroup(U);

			// measurements
			if (i >= params.sweeps / 2)
				p += QCD::ColourWilsonLoops::avgPlaquette(U);
		}

		p /= (params.sweeps + 1) / 2;

		bs.push_back(action.beta);
		plaq.push_back(p);

		// some logging
		if (primaryTask())
			fmt::print("beta = {}, plaq = {}\n", action.beta, p);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(bs, plaq, "plaq");
	}
}
