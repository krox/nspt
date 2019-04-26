#ifndef MODULES_PERT_LANGEVIN_H
#define MODULES_PERT_LANGEVIN_H

#include "modules/module.h"
#include <random>

class MPertLangevinParams
{
  public:
	double eps = 0.05;
	int improvement = 2;
	int reunit = 1;
	int zmreg = 1;
	int gaugefix = 1;
	int count = 1000;
	int seed = -1;
	std::string field;
	std::string filename;
};

class MPertLangevin : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "PertLangevin";
		return name_;
	}

	MPertLangevinParams params;

	MPertLangevin(const json &j)
	{
		j.at("epsilon").get_to(params.eps);
		j.at("reunit").get_to(params.reunit);
		j.at("zmreg").get_to(params.zmreg);
		j.at("gaugefix").get_to(params.gaugefix);
		j.at("improvement").get_to(params.improvement);
		j.at("count").get_to(params.count);
		j.at("field").get_to(params.field);
		j.at("seed").get_to(params.seed);
		if (j.count("filename"))
			j.at("filename").get_to(params.filename);

		if (params.seed == -1)
			params.seed = std::random_device()();
	}

	virtual void run(Environment &env);
};

#endif
