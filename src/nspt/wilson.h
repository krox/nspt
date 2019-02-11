#ifndef NSPT_WILSON_H
#define NSPT_WILSON_H

/**
 * Similar functionality to Grid/qcd/utils/WilsonLoops.h and
 * Grid/qcd/action/gauge/WilsonGaugeAction.h, but:
 *     - uses std::array<LatticeXYZ, 4> instead of LatticeLorentzXYZ
 *     - templated generally enough to support Series<LatticeXYZ>
 */

template <typename Field> double avgPlaquette(const std::array<Field, 4> &U)
{
	double s = 0.0;
	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
			s += sum(trace(U[mu] * Cshift(U[nu], mu, 1) *
			               Cshift(adj(U[mu]), nu, 1) * adj(U[nu])))()()()
			         .real();
	return s * (1.0 / 3.0 / 6.0 / U[0]._grid->gSites());
}

/** derivative of Wilson action (at beta = 1) */
template <typename Field>
void wilsonDeriv(Field &P, const std::array<Field, 4> &U, int mu)
{
	P = 0.0;
	for (int nu = 0; nu < 4; ++nu)
	{
		if (nu == mu)
			continue;
		P += U[mu] * Cshift(U[nu], mu, 1) * Cshift(adj(U[mu]), nu, 1) *
		     adj(U[nu]);
		P += U[mu] * Cshift(Cshift(adj(U[nu]), mu, 1), nu, -1) *
		     Cshift(adj(U[mu]), nu, -1) * Cshift(U[nu], nu, -1);
	}

	P = Ta(P) * (1.0 / 3.0 / 2.0);
}

/** sum of 6 staples */
template <typename Field>
void stapleSum(Field &S, const std::array<Field, 4> &U, int mu)
{
	S = 0.0;
	for (int nu = 0; nu < 4; ++nu)
	{
		if (mu == nu)
			continue;
		S += Cshift(U[nu], mu, 1) * Cshift(adj(U[mu]), nu, 1) * adj(U[nu]);
		S += Cshift(Cshift(adj(U[nu]), mu, 1), nu, -1) *
		     Cshift(adj(U[mu]), nu, -1) * Cshift(U[nu], nu, -1);
	}
}

#endif
