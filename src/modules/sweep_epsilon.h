#ifndef MODULES_SWEEP_EPSILON_H
#define MODULES_SWEEP_EPSILON_H

#include "modules/module.h"
#include <random>

class MSweepEpsilonParams
{
  public:
	// lattice size
	std::vector<int> grid;

	// beta-steps
	double epsilon_min = 0.05;
	double epsilon_max = 0.2;
	int epsilon_steps = 4;

	// evolution parameters
	int sweeps = 100; // sweeps per epsilon-step
	int seed = -1;

	// misc
	bool plot = false;
};

class MSweepEpsilon : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "SweepEpsilon";
		return name_;
	}

	MSweepEpsilonParams params;
	json actionParams;
	json integratorParams;

	MSweepEpsilon(const json &j)
	{
		j.at("grid").get_to(params.grid);
		j.at("epsilon_min").get_to(params.epsilon_min);
		j.at("epsilon_max").get_to(params.epsilon_max);
		j.at("epsilon_steps").get_to(params.epsilon_steps);
		j.at("sweeps").get_to(params.sweeps);
		if (j.count("seed"))
			j.at("seed").get_to(params.seed);
		if (params.seed == -1)
			params.seed = std::random_device()();
		if (j.count("plot"))
			j.at("plot").get_to(params.plot);

		actionParams = j.at("action");
		integratorParams = j.at("integrator");
	}

	virtual void run(Environment &env);
};

#endif
