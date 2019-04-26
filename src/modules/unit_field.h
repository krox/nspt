#ifndef MODULES_UNIT_FIELD_H
#define MODULES_UNIT_FIELD_H

#include "Grid/Grid.h"

#include "modules/module.h"

class MUnitFieldParams
{
  public:
	std::vector<int> grid;
	std::string field_out;
	int order; // 0 for non-perturbative
};

class MUnitField : public Module
{

  public:
	const std::string &name()
	{
		static const std::string name_ = "UnitField";
		return name_;
	}

	MUnitFieldParams params;

	MUnitField(const json &j)
	{
		j.at("grid").get_to(params.grid);
		j.at("field_out").get_to(params.field_out);
		if (j.count("order"))
			j.at("order").get_to(params.order);
		else
			params.order = 0;
	}

	virtual void run(Environment &env);
};

#endif
