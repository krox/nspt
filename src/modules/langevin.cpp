#include "modules/langevin.h"

#include "nspt/grid_utils.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace util;

template <typename GaugeField>
class CompositeAction : public QCD::Action<GaugeField>
{
	std::vector<std::unique_ptr<QCD::Action<GaugeField>>> actions;

  public:
	CompositeAction() = default;
	~CompositeAction() override = default;

	template <typename A, typename... Args> void add(Args &&... args)
	{
		actions.push_back(std::make_unique<A>(std::forward<Args>(args)...));
	}

	void refresh(const GaugeField &U, GridParallelRNG &pRNG) override
	{
		for (auto &a : actions)
			a->refresh(U, pRNG);
	}

	RealD S(const GaugeField &U) override
	{
		RealD s = 0;
		for (auto &a : actions)
			s += a->S(U);
		return s;
	}

	void deriv(const GaugeField &U, GaugeField &dSdU) override
	{
		if (actions.size() == 1)
			actions[0]->deriv(U, dSdU);
		else
		{
			GaugeField tmp(dSdU._grid);
			dSdU = 0.0;

			for (auto &a : actions)
			{
				a->deriv(U, tmp);
				// fmt::print("{}: {}\n", a->action_name(), norm2(tmp));
				dSdU += tmp;
			}
		}
	}

	std::string action_name() override
	{
		std::string r = "[";
		for (auto &a : actions)
			r += a->action_name() + ", ";
		r += "]";
		return r;
	}

	std::string LogParameters() override
	{
		std::string r = "";
		for (auto &a : actions)
			r += a->LogParameters();
		return r;
	}
};

using GaugeField = QCD::LatticeLorentzColourMatrix;
using GaugeMat = QCD::LatticeColourMatrix;
using FermionField = QCD::LatticeSpinColourVector;

/** compute V = exp(aX)U. aliasing is allowed */
[[maybe_unused]] static void evolve(GaugeField &V, double a,
                                    const GaugeField &X, const GaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		QCD::vLorentzColourMatrix tmp = a * X._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY)U. aliasing is allowed */
static void evolve(GaugeField &V, double a, const GaugeField &X, double b,
                   const GaugeField &Y, const GaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		QCD::vLorentzColourMatrix tmp = a * X._odata[ss] + b * Y._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY + cZ))U. aliasing is allowed */
static void evolve(GaugeField &V, double a, const GaugeField &X, double b,
                   const GaugeField &Y, double c, const GaugeField &Z,
                   const GaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		QCD::vLorentzColourMatrix tmp =
		    a * X._odata[ss] + b * Y._odata[ss] + c * Z._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** this creates normal distribution with variance <eta^2>=2 */
static void makeNoise(GaugeField &out, GridParallelRNG &pRNG)
{
	gaussian(pRNG, out);
	out = Ta(out);
}

/** NOTE: this does basic non-rescaled Langevin evolution.
 * For nice correlations, use eps=eps/beta so that the effective drift is
 * invariant of beta */
static void integrateLangevin(GaugeField &U, QCD::Action<GaugeField> &action,
                              GridParallelRNG &pRNG, double eps, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField noise(grid);

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		evolve(U, -eps, force, std::sqrt(eps), noise, U);
	}
}

static void integrateLangevinBF(GaugeField &U, QCD::Action<GaugeField> &action,
                                GridParallelRNG &pRNG, double eps, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField force2(grid);
	GaugeField noise(grid);
	GaugeField Uprime(grid);

	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		// compute force and noise at U
		action.deriv(U, force);
		force = Ta(force);
		makeNoise(noise, pRNG);

		// evolve U' = exp(F) U
		evolve(Uprime, -eps, force, std::sqrt(eps), noise, U);

		// compute force at U'
		action.deriv(Uprime, force2);
		force2 = Ta(force2);

		// evolve U = exp(F') U
		evolve(U, -0.5 * eps, force + force2, std::sqrt(eps), noise,
		       eps * eps * cA / 6.0, force2, U);
	}
}

static void integrateLangevinBauer(GaugeField &U,
                                   QCD::Action<GaugeField> &action,
                                   GridParallelRNG &pRNG, double eps,
                                   int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	GaugeField force(grid);
	GaugeField noise(grid);
	GaugeField Uprime(grid);

	/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
	constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2
	constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2
	constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12

	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		// compute noise and force at U and evolve U' = exp(F) U
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		evolve(Uprime, -eps * k1, force, std::sqrt(eps) * k2, noise, U);

		// compute force at U' and evolve U = exp(F') U
		action.deriv(Uprime, force);
		force = Ta(force);
		evolve(U, -eps - k5 * cA * eps * eps, force, std::sqrt(eps), noise, U);
	}
}

void MLangevin::run(Environment &env)
{
	// gauge field
	GaugeField &U = env.store.get<GaugeField>(params.field);
	GridCartesian *grid = env.getGrid(U._grid->FullDimensions());
	GridRedBlackCartesian *gridRB = env.getGridRB(U._grid->FullDimensions());

	// temporary storage
	GaugeField Uprime(grid);
	GaugeField force(grid);
	GaugeField noise(grid);

	// init RNG
	GridParallelRNG pRNG(grid);
	GridSerialRNG sRNG;
	std::vector<int> pseeds({params.seed});
	pRNG.SeedFixedIntegers(pseeds);

	// track some observables during simulation
	std::vector<double> ts, plaq;

	CompositeAction<GaugeField> action;

	// gauge action
	if (params.gauge_action == "wilson")
		action.add<QCD::WilsonGaugeActionR>(params.beta);
	else if (params.gauge_action == "symanzik")
		action.add<QCD::SymanzikGaugeActionR>(params.beta);
	else
		throw std::runtime_error("unkown gauge action");

	// fermion action
	std::unique_ptr<QCD::WilsonFermion<QCD::WilsonImplR>> fermOperator;
	std::unique_ptr<OperatorFunction<FermionField>> fermSolver;
	double solver_tol = 1.0e-8;
	int solver_max_iter = 5000;
	if (params.fermion_action == "")
	{
	}
	else if (params.fermion_action == "wilson_clover_nf2")
	{
		// NOTE: bare mass is typically negative
		double mass = 0.5 * (1.0 / params.kappa_light - 8.0);
		double csw = params.csw;
		fmt::print("mass = {}, kappa = {}, csw = {}\n", mass,
		           params.kappa_light, csw);

		fermOperator = std::make_unique<QCD::WilsonCloverFermionR>(
		    U, *grid, *gridRB, mass, csw, csw);

		fermSolver = std::make_unique<ConjugateGradient<FermionField>>(
		    solver_tol, solver_max_iter);

		// the action retains references to Fermion and solver. So make sure to
		// keep them alive
		action.add<QCD::TwoFlavourPseudoFermionAction<QCD::WilsonImplR>>(
		    *fermOperator, *fermSolver, *fermSolver);
	}
	else
		throw std::runtime_error("unknown fermion action");

	// rescale step size
	double delta = params.eps / params.beta;

	std::cout << action.LogParameters() << std::endl;

	for (int i = 0; i < params.count; ++i)
	{
		// numerical integration of the langevin process
		if (params.improvement == 0)
			integrateLangevin(U, action, pRNG, delta, params.sweeps);
		else if (params.improvement == 1)
			integrateLangevinBF(U, action, pRNG, delta, params.sweeps);
		else if (params.improvement == 2)
			integrateLangevinBauer(U, action, pRNG, delta, params.sweeps);
		else
			assert(false);

		// project to SU(3) in case of rounding errors
		if (params.reunit)
			ProjectOnGroup(U);

		// measurements
		double p = QCD::ColourWilsonLoops::avgPlaquette(U);
		ts.push_back((i + 1) * params.eps);
		plaq.push_back(p);

		// some logging
		if (primaryTask())
			fmt::print("k = {}/{}, plaq = {}\n", i + 1, params.count, p);

		// write config to file
		if (params.path != "")
		{
			std::string filename =
			    fmt::format("{}/{}{}", params.path, params.prefix, i + 1);
			if (primaryTask())
				fmt::print("writing config to {}\n", filename);
			QCD::NerscIO::writeConfiguration(U, filename, 0, 0);
		}
	}

	if (primaryTask() && params.filename != "")
	{
		fmt::print("writing results to '{}'\n", params.filename);

		auto file = DataFile::create(params.filename);
		file.setAttribute("geom", grid->FullDimensions());
		file.setAttribute("count", params.count);

		file.setAttribute("sweeps", params.sweeps);
		file.setAttribute("eps", params.eps);
		file.setAttribute("improvement", params.improvement);

		file.setAttribute("gauge_action", params.gauge_action);
		file.setAttribute("beta", params.beta);

		file.setAttribute("reunit", params.reunit);

		file.createData("ts", ts);
		file.createData("plaq", plaq);
	}

	if (primaryTask() && params.plot)
	{
		Gnuplot().plotData(plaq, "plaq");
	}
}
