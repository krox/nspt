#ifndef NSPT_WILSON_H
#define NSPT_WILSON_H

#include "nspt/series.h"

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

template <typename Field>
Series<double> avgPlaquette(const std::array<Series<Field>, 4> &U)
{
	int N = (int)U[0].size();
	assert(U[1].size() == N && U[2].size() == N && U[3].size() == N);

	Series<double> s;
	for (int i = 0; i < N; ++i)
		s.append(0.0);

	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
		{
			// TODO: avoid some temp copies
			Series<Field> tmp = U[mu] * Cshift(U[nu], mu, 1) *
			                    Cshift(adj(U[mu]), nu, 1) * adj(U[nu]);
			for (int i = 0; i < N; ++i)
				s[i] += sum(trace(tmp[i]))()()()().real();
		}

	return s * (1.0 / 3.0 / 6.0 / U[0][0]._grid->gSites());
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
		S += Cshift(Cshift(adj(U[nu]), mu, 1) * adj(U[mu]) * U[nu], nu, -1);
	}
}

/** derivative of Wilson action (at beta = 1) */
template <typename Field>
void wilsonDeriv(Field &P, const std::array<Field, 4> &U, int mu)
{
	stapleSum(P, U, mu);
	P = Ta(U[mu] * P);
	P *= 1.0 / 3.0 / 2.0;
}

#endif
