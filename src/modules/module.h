#ifndef MODULES_MODULE_H
#define MODULES_MODULE_H

#include "nlohmann/json.hpp"
#include "util/factory.h"
#include <Grid/Grid.h>
#include <map>
#include <vector>

using nlohmann::json;

class Environment
{
	std::map<std::vector<int>, Grid::GridCartesian *> grids_;
	std::map<std::vector<int>, Grid::GridRedBlackCartesian *> gridsRB_;

  public:
	// default-constructable but not copyable
	Environment() {}
	Environment(const Environment &) = delete;

	// global object storage for gauge-fields, propagators and such
	util::Store store;

	// retrieve grid of specific size or create a new one
	// NOTE: these are cached and never deleted
	Grid::GridCartesian *getGrid(const std::vector<int> &);
	Grid::GridRedBlackCartesian *getGridRB(const std::vector<int> &);
};

class Module
{
  public:
	virtual void run(Environment &env) = 0;
	virtual ~Module() {}
	virtual const std::string &name() { assert(false); };
};

std::unique_ptr<Module> createModule(const std::string &id, const json &params);

#endif
