#ifndef MODULES_UNIT_FIELD_H
#define MODULES_UNIT_FIELD_H

#include "Grid/Grid.h"

#include "modules/module.h"

class MUnitFieldParams
{
  public:
	std::vector<int> grid;
	std::string field_out;
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
	}

	virtual void run(Environment &env);
};

#endif
