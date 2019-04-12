#ifndef MODULES_RANDOM_FIELD_H
#define MODULES_RANDOM_FIELD_H

#include "Grid/Grid.h"

#include "modules/module.h"
#include <random>

class MURandomFieldParams
{
  public:
	std::vector<int> grid;
	std::string field_out;
	int seed = -1;
};

class MRandomField : public Module
{

  public:
	const std::string &name()
	{
		static const std::string name_ = "RandomField";
		return name_;
	}

	MURandomFieldParams params;

	MRandomField(const json &j)
	{
		j.at("grid").get_to(params.grid);
		j.at("field_out").get_to(params.field_out);
		j.at("seed").get_to(params.seed);
		if (params.seed == -1)
			params.seed = std::random_device()();
	}

	virtual void run(Environment &env);
};

#endif
