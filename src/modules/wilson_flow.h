#ifndef MODULES_WILSON_FLOW_H
#define MODULES_WILSON_FLOW_H

#include "Grid/Grid.h"

#include "modules/module.h"

class MWilsonFlowParams
{
  public:
	std::string field;
	double t_max = 3.0;
	double t_step = 0.2;
	bool plot = false;
};

class MWilsonFlow : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "WilsonFlow";
		return name_;
	}

	MWilsonFlowParams params;

	MWilsonFlow(const json &j)
	{
		j.at("field").get_to(params.field);
		j.at("t_max").get_to(params.t_max);
		j.at("t_step").get_to(params.t_step);
		if (j.count("plot"))
			j.at("plot").get_to(params.plot);
	}

	virtual void run(Environment &env);
};

#endif
