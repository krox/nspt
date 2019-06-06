#ifndef MODULES_HMC_H
#define MODULES_HMC_H

#include "modules/module.h"
#include <random>

class MHMCParams
{
  public:
	// IO
	std::string field;
	std::string filename = ""; // one hdf5 with plaquette and such
	std::string path = "";     // path to store configs in
	std::string prefix = "";   // prefix for config files

	// HMC integration
	double eps = 0.05;
	bool metropolis = true;
	int count = 1000;
	int seed = -1;
	int sweeps = 1;
	std::string rng = ""; // file to read rng state from

	// misc
	bool plot = false;
};

class MHMC : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "HMC";
		return name_;
	}

	MHMCParams params;
	json actionParams;

	MHMC(const json &j)
	{
		// io params
		j.at("field").get_to(params.field);
		if (j.count("filename"))
			j.at("filename").get_to(params.filename);
		if (j.count("path"))
			j.at("path").get_to(params.path);
		if (j.count("prefix"))
			j.at("prefix").get_to(params.prefix);

		// langevin params
		j.at("epsilon").get_to(params.eps);
		j.at("metropolis").get_to(params.metropolis);
		j.at("count").get_to(params.count);
		if (j.count("seed"))
			j.at("seed").get_to(params.seed);
		if (params.seed == -1)
			params.seed = std::random_device()();
		j.at("sweeps").get_to(params.sweeps);
		if (j.count("rng"))
			j.at("rng").get_to(params.rng);

		// gauge action
		actionParams = j.at("action");

		// misc
		if (j.count("plot"))
			j.at("plot").get_to(params.plot);
	}

	virtual void run(Environment &env);
};

#endif
