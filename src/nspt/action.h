#ifndef NSPT_ACTION_H
#define NSPT_ACTION_H

#include "nspt/grid_utils.h"

using namespace Grid;
using namespace util;

template <typename GaugeField>
class CompositeAction : public QCD::Action<GaugeField>
{
	using FermionField = QCD::LatticeSpinColourVector;

	// gauge action
	std::unique_ptr<QCD::Action<GaugeField>> gaugeAction;

	// fermion action
	std::unique_ptr<QCD::Action<GaugeField>> fermAction;
	std::unique_ptr<QCD::WilsonFermion<QCD::WilsonImplR>> fermOperator;
	std::unique_ptr<OperatorFunction<FermionField>> fermSolver;

  public:
	std::string gauge_action = "";
	double beta = 0.0;
	std::string fermion_action = "";
	double kappa_light = 0.0;
	double csw = 0.0;

	CompositeAction(const json &j, GridCartesian *grid,
	                GridRedBlackCartesian *gridRB)
	{
		// read parameters
		j.at("gauge_action").get_to(gauge_action);
		j.at("beta").get_to(beta);
		if (j.count("fermion_action"))
			j.at("fermion_action").get_to(fermion_action);
		if (j.count("kappa_light"))
			j.at("kappa_light").get_to(kappa_light);
		if (j.count("csw"))
			j.at("csw").get_to(csw);

		// gauge action
		if (gauge_action == "wilson")
			gaugeAction = std::make_unique<QCD::WilsonGaugeActionR>(beta);
		else if (gauge_action == "symanzik")
			gaugeAction = std::make_unique<QCD::SymanzikGaugeActionR>(beta);
		else
			throw std::runtime_error("unkown gauge action");
		std::cout << gaugeAction->LogParameters() << std::endl;

		// fermion action
		if (fermion_action == "")
		{
			// quenched
		}
		else if (fermion_action == "wilson_clover_nf2")
		{
			using Fermion = QCD::WilsonCloverFermionR;
			using Solver = ConjugateGradient<FermionField>;
			using FermAction =
			    QCD::TwoFlavourPseudoFermionAction<QCD::WilsonImplR>;

			// NOTE: bare mass is typically negative
			double mass = 0.5 * (1.0 / kappa_light - 8.0);

			// solver parameters
			double solver_tol = 1.0e-11;
			int solver_max_iter = 5000;

			if (primaryTask())
				fmt::print("mass = {}, kappa = {}, csw = {}\n", mass,
				           kappa_light, csw);

			GaugeField U(grid);
			U = 1.0; // NOTE: has to be initialized, or the constructor of
			         // fermOp crashes

			fermOperator =
			    std::make_unique<Fermion>(U, *grid, *gridRB, mass, csw, csw);
			fermSolver = std::make_unique<Solver>(solver_tol, solver_max_iter);
			fermAction = std::make_unique<FermAction>(*fermOperator,
			                                          *fermSolver, *fermSolver);
		}
		else
			throw std::runtime_error("unknown fermion action");
	}

	void refresh(const GaugeField &U, GridParallelRNG &pRNG) override
	{
		if (gaugeAction)
			gaugeAction->refresh(U, pRNG); // probably does nothing
		if (fermAction)
			fermAction->refresh(U, pRNG);
	}

	RealD S(const GaugeField &U) override
	{
		RealD s = 0;
		if (gaugeAction)
			s += gaugeAction->S(U);
		if (fermAction)
			s += fermAction->S(U);
		return s;
	}

	void deriv(const GaugeField &U, GaugeField &dSdU) override
	{
		assert(gaugeAction);
		gaugeAction->deriv(U, dSdU);
		// double n = norm2(dSdU) / U._grid->gSites();
		// if (primaryTask())
		//	fmt::print("{} force = {}\n", gaugeAction->action_name(), n);

		if (fermAction)
		{
			GaugeField tmp(dSdU._grid);
			fermAction->deriv(U, tmp);
			// n = norm2(tmp) / U._grid->gSites();
			// if (primaryTask())
			//	fmt::print("{} force = {}\n", fermAction->action_name(), n);
			dSdU += tmp;
		}
	}

	std::string action_name() override
	{
		assert(gaugeAction);
		std::string r = gaugeAction->action_name();
		if (fermAction)
		{
			r += " ";
			r += fermAction->action_name();
		}
		return r;
	}

	std::string LogParameters() override
	{
		std::string r = "";
		if (gaugeAction)
			r += gaugeAction->LogParameters();
		if (fermAction)
			r += fermAction->LogParameters();
		return r;
	}
};

#endif
