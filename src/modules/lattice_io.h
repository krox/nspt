#ifndef MODULES_LATTICE_IO_H
#define MODULES_LATTICE_IO_H

#include "Grid/Grid.h"

#include "modules/module.h"

class MWriteFieldParams
{
  public:
	std::string field;
	std::string filename;
	std::string format;
};

class MWriteField : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "WriteField";
		return name_;
	}

	MWriteFieldParams params;

	MWriteField(const json &j)
	{
		j.at("field").get_to(params.field);
		j.at("filename").get_to(params.filename);
		if (j.count("format"))
			j.at("format").get_to(params.format);
		else
			params.format = "";
	}

	virtual void run(Environment &env);
};

class MReadFieldParams
{
  public:
	std::string field;
	std::string filename;
	std::string format;
	std::vector<int> grid; // only used for nersc format right now
};

class MReadField : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "ReadField";
		return name_;
	}

	MReadFieldParams params;

	MReadField(const json &j)
	{
		j.at("field").get_to(params.field);
		j.at("filename").get_to(params.filename);
		if (j.count("format"))
			j.at("format").get_to(params.format);
		else
			params.format = "";
		if (j.count("grid"))
			j.at("grid").get_to(params.grid);
	}

	virtual void run(Environment &env);
};

#endif
