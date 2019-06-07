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

void integrateLangevin(LatticeGaugeField &U,
                       CompositeAction<LatticeGaugeField> &action,
                       GridSerialRNG &, GridParallelRNG &pRNG, double eps,
                       int sweeps, double &plaq, double &loop)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField noise(grid);

	plaq = 0.0;
	loop = 0.0;
	for (int i = 0; i < sweeps; ++i)
	{
		action.refresh(U, pRNG);
		makeNoise(noise, pRNG);
		action.deriv(U, force);
		force = Ta(force);
		evolve(U, -eps, force, std::sqrt(eps), noise, U);

		ProjectOnGroup(U);
		plaq += QCD::ColourWilsonLoops::avgPlaquette(U);
		loop += real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
	}
	plaq /= sweeps;
	loop /= sweeps;
}

void integrateLangevinBF(LatticeGaugeField &U,
                         CompositeAction<LatticeGaugeField> &action,
                         GridSerialRNG &, GridParallelRNG &pRNG, double eps,
                         int sweeps, double &plaq, double &loop)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField force2(grid);
	LatticeGaugeField noise(grid);
	LatticeGaugeField Uprime(grid);

	double cA = 3.0; // = Nc = casimir in adjoint representation

	plaq = 0.0;
	loop = 0.0;
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

		ProjectOnGroup(U);
		plaq += QCD::ColourWilsonLoops::avgPlaquette(U);
		loop += real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
	}
	plaq /= sweeps;
	loop /= sweeps;
}

void integrateLangevinBauer(LatticeGaugeField &U,
                            CompositeAction<LatticeGaugeField> &action,
                            GridSerialRNG &, GridParallelRNG &pRNG, double eps,
                            int sweeps, double &plaq, double &loop)
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

	plaq = 0.0;
	loop = 0.0;
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

		ProjectOnGroup(U);
		plaq += QCD::ColourWilsonLoops::avgPlaquette(U);
		loop += real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
	}
	plaq /= sweeps;
	loop /= sweeps;
}

void integrateHMC(LatticeGaugeField &U,
                  CompositeAction<LatticeGaugeField> &action, GridSerialRNG &,
                  GridParallelRNG &pRNG, double eps, int sweeps, double &plaq,
                  double &loop)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeGaugeField force(grid);
	LatticeGaugeField mom(grid);

	plaq = 0.0;
	loop = 0.0;
	for (int iter = 0; iter < sweeps; ++iter)
	{
		// new pseudo-fermions and momenta
		action.refresh(U, pRNG);
		gaussian(pRNG, mom);
		mom = std::sqrt(0.5) * Ta(mom); // TODO: right scale here?

		// leap-frog integration
		QCD::evolve(U, 0.5 * eps, mom, U);
		action.deriv(U, force);
		force *= -eps;
		mom += force;
		QCD::evolve(U, 0.5 * eps, mom, U);

		ProjectOnGroup(U);
		plaq += QCD::ColourWilsonLoops::avgPlaquette(U);
		loop += real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
	}
	plaq /= sweeps;
	loop /= sweeps;
}

void quenchedHeatbath(LatticeGaugeField &U,
                      CompositeAction<LatticeGaugeField> &action,
                      GridSerialRNG &sRNG, GridParallelRNG &pRNG, double,
                      int sweeps, double &plaq, double &loop)
{
	conformable(U._grid, pRNG._grid);
	auto grid = U._grid;
	LatticeColourMatrix link(grid);
	LatticeColourMatrix staple(grid);

	// init checkerboard
	QCD::LatticeInteger mask(grid);
	parallel_for(int ss = 0; ss < grid->oSites(); ++ss)
	{
		std::vector<int> co;
		grid->oCoorFromOindex(co, ss);
		int s = 0;
		for (int mu = 0; mu < QCD::Nd; ++mu)
			s += co[mu];
		mask._odata[ss] = s % 2;
	}

	plaq = 0.0;
	loop = 0.0;
	for (int k = 0; k < sweeps; ++k)
	{
		for (int cb = 0; cb < 2; ++cb, mask = Integer(1) - mask)
			for (int mu = 0; mu < QCD::Nd; ++mu)
			{
				ColourWilsonLoops::Staple(staple, U, mu);
				link = peekLorentz(U, mu);

				for (int sg = 0; sg < SU3::su2subgroups(); sg++)
					SU3::SubGroupHeatBath(sRNG, pRNG, action.beta, link, staple,
					                      sg, 20, mask);

				pokeLorentz(U, link, mu);
			}

		ProjectOnGroup(U);
		plaq += QCD::ColourWilsonLoops::avgPlaquette(U);
		loop += real(QCD::ColourWilsonLoops::avgPolyakovLoop(U));
	}
	plaq /= sweeps;
	loop /= sweeps;
}

} // namespace QCD
} // namespace Grid
