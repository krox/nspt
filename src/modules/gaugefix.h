#ifndef MODULES_GAUGEFIX_H
#define MODULES_GAUGEFIX_H

#include "Grid/Grid.h"

#include "modules/module.h"

class GaugefixIntegrator
{
	// shorthands. possibly template paramters in the future
	using GaugeField = Grid::QCD::LatticeLorentzColourMatrix;
	using GaugeMat = Grid::QCD::LatticeColourMatrix;
	using LatticeReal = Grid::QCD ::LatticeReal;

	// temporaries
	mutable GaugeMat R, tmp;

	// Fourier-acceleration
	LatticeReal prec;
	Grid::FFT fft;

  public:
	GaugefixIntegrator(Grid::GridCartesian *grid);
	double gaugecond(const GaugeField &U) const;
	void step(GaugeField &U, double eps);
};

class MGaugefixParams
{
  public:
	std::string field;
	int iter_max = 1000;
	double gcond = 1.0e-8;
	bool error_on_fail = true;
};

class MGaugefix : public Module
{
  public:
	const std::string &name()
	{
		static const std::string name_ = "Gaugefix";
		return name_;
	}

	MGaugefixParams params;

	MGaugefix(const json &j)
	{
		j.at("field").get_to(params.field);
		j.at("iter_max").get_to(params.iter_max);
		j.at("gcond").get_to(params.gcond);
		if (j.count("error_on_fail"))
			j.at("error_on_fail").get_to(params.error_on_fail);
	}

	virtual void run(Environment &env);
};

#endif
