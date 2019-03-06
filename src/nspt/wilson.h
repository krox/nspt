#ifndef NSPT_WILSON_H
#define NSPT_WILSON_H

/**
 * Similar functionality to Grid/qcd/utils/WilsonLoops.h and
 * Grid/qcd/action/gauge/WilsonGaugeAction.h, but:
 *     - uses std::array<LatticeXYZ, 4> instead of LatticeLorentzXYZ
 *     - Support Series types
 */

double avgPlaquette(const std::array<LatticeColourMatrix, 4> &U)
{
	double s = 0.0;
	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
			s += sum(trace(U[mu] * Cshift(U[nu], mu, 1) *
			               Cshift(adj(U[mu]), nu, 1) * adj(U[nu])))()()()()
			         .real();
	return s * (1.0 / 3.0 / 6.0 / U[0]._grid->gSites());
}

RealSeries avgPlaquette(const std::array<LatticeColourMatrixSeries, 4> &U)
{
	RealSeries s = 0.0;
	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
			s += real(sum(trace(U[mu] * Cshift(U[nu], mu, 1) *
			                    Cshift(adj(U[mu]), nu, 1) * adj(U[nu]))));
	return s * (1.0 / 3.0 / 6.0 / U[0]._grid->gSites());
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
		S += Cshift(U[nu], mu, 1) * adj(Cshift(U[mu], nu, 1)) * adj(U[nu]);
		S += Cshift(Field(adj(Cshift(U[nu], mu, 1)) * adj(U[mu]) * U[nu]), nu,
		            -1);
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
