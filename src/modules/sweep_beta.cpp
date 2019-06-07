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

	// choose integration scheme
	using Integrator =
	    void (*)(GaugeField &, CompositeAction<GaugeField> &, GridSerialRNG &,
	             GridParallelRNG &, double, int, double &, double &);
	Integrator integrator = nullptr;
	if (params.method == "LangevinEuler")
		integrator = &QCD::integrateLangevin;
	else if (params.method == "LangevinBF")
		integrator = &QCD::integrateLangevinBF;
	else if (params.method == "LangevinBauer")
		integrator = &QCD::integrateLangevinBauer;
	else if (params.method == "HMC")
		integrator = &QCD::integrateHMC;
	else if (params.method == "heatbath")
	{
		assert(actionParams.at("gauge_action") == "wilson");
		assert(actionParams.at("fermion_action") == "");
		integrator = &QCD::quenchedHeatbath;
	}
	else
		throw std::runtime_error("unknown integrator");

	// track some observables during simulation
	std::vector<double> bs, plaq, loop;

	for (int k = 0; k < params.beta_steps; ++k)
	{
		// create action (NOTE: dont do beta=0. bad for rescaling)
		actionParams["beta"] = params.beta_max / params.beta_steps * (k + 1);
		CompositeAction<GaugeField> action(actionParams, grid, gridRB);
		std::cout << action.LogParameters() << std::endl;

		// rescale step size
		double delta = 0.0 / 0.0;
		if (params.method == "HMC")
			delta = params.eps / std::sqrt(action.beta);
		else
			delta = params.eps / action.beta;

		double p, l;

		// first half of sweeps for thermalization
		integrator(U, action, sRNG, pRNG, delta, params.sweeps / 2, p, l);

		// second half with measurements
		integrator(U, action, sRNG, pRNG, delta, params.sweeps / 2, p, l);

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
