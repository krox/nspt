#ifndef MODULES_MESON_2PT_H
#define MODULES_MESON_2PT_H

#include "modules/module.h"

class MMeson2ptParams
{
  public:
	// IO
	std::string field;
	std::string filename;

	// fermion action
	std::string fermion_action; // wilson_clover_nf2
	double csw = 0.0;
	double kappa_light = 0.0;

	std::vector<int> source;
	std::vector<std::vector<int>> sink_mom_list;
};

class MMeson2pt : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "Meson2pt";
		return name_;
	}

	MMeson2ptParams params;

	MMeson2pt(const json &j)
	{
		j.at("field").get_to(params.field);
		j.at("filename").get_to(params.filename);
		j.at("fermion_action").get_to(params.fermion_action);
		j.at("csw").get_to(params.csw);
		j.at("kappa_light").get_to(params.kappa_light);
		j.at("source").get_to(params.source);
		j.at("sink_mom_list").get_to(params.sink_mom_list);
	}

	virtual void run(Environment &env);
};

#endif
