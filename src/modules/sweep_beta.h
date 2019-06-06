#ifndef MODULES_SWEEP_BETA_H
#define MODULES_SWEEP_BETA_H

#include "modules/module.h"
#include <random>

class MSweepBetaParams
{
  public:
	// lattice size
	std::vector<int> grid;

	// beta-steps
	double beta_max = 6.0;
	int beta_steps = 21;

	// evolution parameters
	std::string method; // LangevinEuler/BF/Bauer, HMC
	double eps = 0.05;
	int sweeps = 10; // sweeps per beta-step
	int seed = -1;

	// misc
	bool plot = false;
};

class MSweepBeta : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "SweepBeta";
		return name_;
	}

	MSweepBetaParams params;
	json actionParams;

	MSweepBeta(const json &j)
	{
		j.at("grid").get_to(params.grid);
		j.at("epsilon").get_to(params.eps);
		j.at("beta_max").get_to(params.beta_max);
		j.at("beta_steps").get_to(params.beta_steps);
		j.at("method").get_to(params.method);
		j.at("sweeps").get_to(params.sweeps);
		if (j.count("seed"))
			j.at("seed").get_to(params.seed);
		if (params.seed == -1)
			params.seed = std::random_device()();
		if (j.count("plot"))
			j.at("plot").get_to(params.plot);

		actionParams = j.at("action");
	}

	virtual void run(Environment &env);
};

#endif
