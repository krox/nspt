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

template <typename Field>
Series<double> avgPlaquette(const std::array<Series<Field>, 4> &U)
{
	int N = (int)U[0].size();
	assert(U[1].size() == N && U[2].size() == N && U[3].size() == N);

	Series<double> s;
	for (int i = 0; i < N; ++i)
		s.append(0.0);

	// FIXME: not optimal
	for (int a = 0; a < N; ++a)
		for (int b = 0; b < N - a; ++b)
			for (int c = 0; c < N - a - b; ++c)
				for (int d = 0; d < N - a - b - c; ++d)
					for (int mu = 0; mu < 4; ++mu)
						for (int nu = mu + 1; nu < 4; ++nu)
							s[a + b + c + d] +=
							    sum(trace(U[mu][a] * Cshift(U[nu][b], mu, 1) *
							              Cshift(adj(U[mu][c]), nu, 1) *
							              adj(U[nu][d])))()()()
							        .real();

	return s * (1.0 / 3.0 / 6.0 / U[0][0]._grid->gSites());
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
