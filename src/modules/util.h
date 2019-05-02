#ifndef MODULES_UTIL_H
#define MODULES_UTIL_H

#include "Grid/Grid.h"

#include "modules/module.h"

class MDeleteObjectParams
{
  public:
	std::string name;
};

class MDeleteObject : public Module
{

  public:
	const std::string &name()
	{
		static const std::string name_ = "DeleteObject";
		return name_;
	}

	MDeleteObjectParams params;

	MDeleteObject(const json &j) { j.at("name").get_to(params.name); }

	virtual void run(Environment &env);
};

#endif
