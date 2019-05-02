#ifndef NSPT_WILSON_H
#define NSPT_WILSON_H

#include "nspt/pqcd.h"
#include <Grid/Grid.h>
using namespace Grid;

/**
 * Similar functionality to Grid/qcd/utils/WilsonLoops.h and
 * Grid/qcd/action/gauge/WilsonGaugeAction.h, but:
 *     - uses std::array<LatticeXYZ, 4> instead of LatticeLorentzXYZ
 *     - Support Series types
 */

inline double avgPlaquette(const std::array<QCD::LatticeColourMatrix, 4> &U,
                           int mu, int nu)
{
	double s = sum(trace(U[mu] * Cshift(U[nu], mu, 1) *
	                     Cshift(adj(U[mu]), nu, 1) * adj(U[nu])))()()()
	               .real();
	return s * (1.0 / QCD::Nc / U[0]._grid->gSites());
}

inline double avgPlaquette(const std::array<QCD::LatticeColourMatrix, 4> &U)
{
	double s = 0.0;
	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
			s += avgPlaquette(U, mu, nu);
	return s * (1.0 / 6.0);
}

inline pQCD::RealSeries
avgPlaquette(const std::array<pQCD::LatticeColourMatrixSeries, 4> &U, int mu,
             int nu)
{
	pQCD::RealSeries s =
	    real(sum(trace(U[mu] * Cshift(U[nu], mu, 1) *
	                   Cshift(adj(U[mu]), nu, 1) * adj(U[nu]))));
	return s * (1.0 / pQCD::Nc / U[0]._grid->gSites());
}

inline pQCD::RealSeries
avgPlaquette(const std::array<pQCD::LatticeColourMatrixSeries, 4> &U)
{
	pQCD::RealSeries s = 0.0;
	for (int mu = 0; mu < 4; ++mu)
		for (int nu = mu + 1; nu < 4; ++nu)
			s += avgPlaquette(U, mu, nu);
	return s * (1.0 / 6.0);
}

/** sum of 6 staples */
template <typename Field>
void stapleSum(Field &S, const std::array<Field, 4> &U, int mu)
{
	// NOTE: This function is the bottleneck in NSPT Langevin evolution.
	//       Maybe some lowlevel optimization is still possible.

	S = 0.0;
	for (int nu = 0; nu < 4; ++nu)
	{
		if (mu == nu)
			continue;
		Field tmp = Cshift(U[nu], mu, 1);
		S += tmp * adj(Cshift(U[mu], nu, 1)) * adj(U[nu]);
		S += Cshift(Field(adj(tmp) * adj(U[mu]) * U[nu]), nu, -1);
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
