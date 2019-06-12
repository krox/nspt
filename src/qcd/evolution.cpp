#include "qcd/evolution.h"

#include "fmt/format.h"
#include "nspt/grid_utils.h"

namespace Grid {
namespace QCD {

/** compute V = exp(aX)U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp = a * X._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY)U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            double b, const LatticeGaugeField &Y, const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp = a * X._odata[ss] + b * Y._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

/** compute V = exp(aX +bY + cZ))U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            double b, const LatticeGaugeField &Y, double c,
            const LatticeGaugeField &Z, const LatticeGaugeField &U)
{
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vLorentzColourMatrix tmp =
		    a * X._odata[ss] + b * Y._odata[ss] + c * Z._odata[ss];
		for (int mu = 0; mu < 4; ++mu)
			V._odata[ss](mu) = Exponentiate(tmp(mu), 1.0) * U._odata[ss](mu);
	}
}

void makeNoise(LatticeGaugeField &out, GridParallelRNG &pRNG)
{
	gaussian(pRNG, out);
	out = Ta(out);
	auto n = norm2(out) / out._grid->gSites();
	if (primaryTask())
		fmt::print("noise force = {}\n", n);
}

} // namespace QCD
} // namespace Grid

using namespace Grid;
using namespace QCD;

void LangevinEuler::run(LatticeGaugeField &U, GridSerialRNG &,
                        GridParallelRNG &pRNG, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField noise(grid);

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		evolve(U, -delta, force, std::sqrt(delta), noise, U);

		ProjectOnGroup(U);
		trackObservables(U);
	}
}

void LangevinBF::run(LatticeGaugeField &U, GridSerialRNG &,
                     GridParallelRNG &pRNG, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField force2(grid);
	LatticeGaugeField noise(grid);
	LatticeGaugeField Uprime(grid);

	double cA = 3.0; // = Nc = casimir in adjoint representation

	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		// compute force and noise at U
		action.deriv(U, force);
		force = Ta(force);
		makeNoise(noise, pRNG);

		// evolve U' = exp(F) U
		evolve(Uprime, -delta, force, std::sqrt(delta), noise, U);

		// compute force at U'
		action.deriv(Uprime, force2);
		force2 = Ta(force2);

		// evolve U = exp(F') U
		evolve(U, -0.5 * delta, force + force2, std::sqrt(delta), noise,
		       delta * delta * cA / 6.0, force2, U);

		ProjectOnGroup(U);
		trackObservables(U);
	}
}

void LangevinBauer::run(LatticeGaugeField &U, GridSerialRNG &,
                        GridParallelRNG &pRNG, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField noise(grid);
	LatticeGaugeField Uprime(grid);

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
		evolve(Uprime, -delta * k1, force, std::sqrt(delta) * k2, noise, U);

		// compute force at U' and evolve U = exp(F') U
		action.deriv(Uprime, force);
		force = Ta(force);
		evolve(U, -delta - k5 * cA * delta * delta, force, std::sqrt(delta),
		       noise, U);

		ProjectOnGroup(U);
		trackObservables(U);
	}
}

void HMC::run(LatticeGaugeField &U, GridSerialRNG &, GridParallelRNG &pRNG,
              int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField mom(grid);

	for (int iter = 0; iter < sweeps; ++iter)
	{
		// new pseudo-fermions and momenta
		action.refresh(U, pRNG);
		gaussian(pRNG, mom);
		mom = std::sqrt(0.5) * Ta(mom); // TODO: right scale here?

		// leap-frog integration
		evolve(U, 0.5 * delta, mom, U);
		action.deriv(U, force);
		force *= -delta;
		mom += force;
		evolve(U, 0.5 * delta, mom, U);

		ProjectOnGroup(U);
		trackObservables(U);
	}
}

void Heatbath::run(LatticeGaugeField &U, GridSerialRNG &sRNG,
                   GridParallelRNG &pRNG, int sweeps)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeColourMatrix link(grid);
	LatticeColourMatrix staple(grid);

	// init checkerboard
	LatticeInteger mask(grid);
	parallel_for(int ss = 0; ss < grid->oSites(); ++ss)
	{
		std::vector<int> co;
		grid->oCoorFromOindex(co, ss);
		int s = 0;
		for (int mu = 0; mu < Nd; ++mu)
			s += co[mu];
		mask._odata[ss] = s % 2;
	}

	for (int k = 0; k < sweeps; ++k)
	{
		for (int cb = 0; cb < 2; ++cb, mask = Integer(1) - mask)
			for (int mu = 0; mu < Nd; ++mu)
			{
				ColourWilsonLoops::Staple(staple, U, mu);
				link = peekLorentz(U, mu);

				for (int sg = 0; sg < SU3::su2subgroups(); sg++)
					SU3::SubGroupHeatBath(sRNG, pRNG, beta, link, staple, sg,
					                      20, mask);

				pokeLorentz(U, link, mu);
			}

		ProjectOnGroup(U);
		trackObservables(U);
	}
}

std::unique_ptr<QCDIntegrator> makeQCDIntegrator(QCDIntegrator::Action &action,
                                                 const json &j)
{
	auto method = j.at("method").get<std::string>();
	if (method == "LangevinEuler")
		return std::make_unique<LangevinEuler>(action,
		                                       j.at("epsilon").get<double>());
	if (method == "LangevinBF")
		return std::make_unique<LangevinBF>(action,
		                                    j.at("epsilon").get<double>());
	if (method == "LangevinBauer")
		return std::make_unique<LangevinBauer>(action,
		                                       j.at("epsilon").get<double>());
	if (method == "HMC")
		return std::make_unique<HMC>(action, j.at("epsilon").get<double>(),
		                             j.at("metropolis").get<bool>());
	if (method == "Heatbath")
		return std::make_unique<Heatbath>(action);
	throw std::runtime_error("unknown integrator");
}
