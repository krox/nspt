#ifndef MODULES_MODULE_H
#define MODULES_MODULE_H

#include "nlohmann/json.hpp"
#include "util/factory.h"

using nlohmann::json;

class Environment
{
  public:
	// default-constructable but not copyable
	Environment() {}
	Environment(const Environment &) = delete;

	util::Store store; // global object storage
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
