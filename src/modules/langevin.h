#ifndef MODULES_LANGEVIN_H
#define MODULES_LANGEVIN_H

#include "modules/module.h"
#include <random>

class MLangevinParams
{
  public:
	// IO
	std::string field;
	std::string filename = ""; // one hdf5 with plaquette and such
	std::string path = "";     // path to store configs in
	std::string prefix = "";   // prefix for config files

	// Langevin integration
	double eps = 0.05;
	int improvement = 2;
	int count = 1000;
	int seed = -1;
	int sweeps = 1;
	int reunit = 1;
	std::string rng = ""; // file to read rng state from

	// misc
	bool plot = false;
};

class MLangevin : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "Langevin";
		return name_;
	}

	MLangevinParams params;
	json actionParams;

	MLangevin(const json &j)
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
		j.at("improvement").get_to(params.improvement);
		j.at("count").get_to(params.count);
		if (j.count("seed"))
			j.at("seed").get_to(params.seed);
		if (params.seed == -1)
			params.seed = std::random_device()();
		j.at("sweeps").get_to(params.sweeps);
		j.at("reunit").get_to(params.reunit);
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
