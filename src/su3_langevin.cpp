#include "Grid/Grid.h"

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

#include "nspt/wilson.h"

using namespace util;

/** Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	GridCartesian *grid;
	GridRedBlackCartesian *rbGrid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// gauge config
	std::array<LatticeColourMatrix, 4> U;

	// temporaries
	std::array<LatticeColourMatrix, 4> force, force2;
	LatticeColourMatrix Umu, drift, rot, tmp1, tmp2;

	/** NOTE: does not initialize gauge field */
	Langevin(std::vector<int> latt)
	    : grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      rbGrid(SpaceTimeGrid::makeFourDimRedBlackGrid(grid)),
	      pRNG(grid), U{LatticeColourMatrix(grid), LatticeColourMatrix(grid),
	                    LatticeColourMatrix(grid), LatticeColourMatrix(grid)},
	      force{LatticeColourMatrix(grid), LatticeColourMatrix(grid),
	            LatticeColourMatrix(grid), LatticeColourMatrix(grid)},
	      force2{LatticeColourMatrix(grid), LatticeColourMatrix(grid),
	             LatticeColourMatrix(grid), LatticeColourMatrix(grid)},
	      Umu(grid), drift(grid), rot(grid), tmp1(grid), tmp2(grid)
	{
		std::vector<int> pseeds({1, 2, 3, 4, 5});
		std::vector<int> sseeds({6, 7, 8, 9, 10});
		pRNG.SeedFixedIntegers(pseeds);
		sRNG.SeedFixedIntegers(sseeds);
	}

	void makeNoise(LatticeColourMatrix &out, double eps)
	{
		// this iscorrect for abelian groups
		SU3::GaussianFundamentalLieAlgebraMatrix(pRNG, out, eps);

		// naive improvement (works, but seems not significant)
		// Idea: split eps in two parts, and use BCH formula to multiply
		/*SU3::GaussianFundamentalLieAlgebraMatrix(pRNG, tmp1,
		                                         std::sqrt(0.5) * eps);
		SU3::GaussianFundamentalLieAlgebraMatrix(pRNG, tmp2,
		                                         std::sqrt(0.5) * eps);
		out = tmp1 + tmp2 + 0.5 * (tmp1 * tmp2 - tmp2 * tmp1);*/
	}

	/** 1st order method (forward Euler) */
	void evolveStep(RealD beta, double eps)
	{
		// force at t=0
		for (int mu = 0; mu < 4; ++mu)
			wilsonDeriv(force[mu], U, mu);

		// evolve with force + drift to t=eps
		for (int mu = 0; mu < Nd; ++mu)
		{
			makeNoise(drift, std::sqrt(2.0 * eps));
			drift += (-eps * beta) * force[mu];
			SU3::taExp(drift, rot);
			U[mu] = rot * U[mu];
		}

		// ProjectOnGroup(U);
	}

	/** second order method (Heun method) */
	void evolveStepImproved(RealD beta, double eps)
	{
		// force at t=0
		for (int mu = 0; mu < 4; ++mu)
			wilsonDeriv(force[mu], U, mu);

		// evolve with force + drift to t=eps
		for (int mu = 0; mu < Nd; ++mu)
		{
			makeNoise(drift, std::sqrt(2.0 * eps));
			drift += (-eps * beta) * force[mu];
			SU3::taExp(drift, rot);
			U[mu] = rot * U[mu];
		}

		// force at new point
		for (int mu = 0; mu < 4; ++mu)
			wilsonDeriv(force2[mu], U, mu);

		// correct to 0.5*(force + force2)
		for (int mu = 0; mu < Nd; ++mu)
		{
			drift = (-eps * beta * 0.5) * (force2[mu] - force[mu]);
			SU3::taExp(drift, rot);
			U[mu] = rot * U[mu];
		}

		// ProjectOnGroup(U);
	}
};

int main(int argc, char **argv)
{
	Grid_init(&argc, &argv);

	auto lang = Langevin(GridDefaultLatt());

	auto plot = Gnuplot();
	plot.style = "lines";

	double beta = 6.0;
	double maxT = 10.0;

	for (double eps = 0.1; eps > 0.01; eps /= 2.0)
	{
		std::vector<double> xs, ys;

		lang.U[0] = 1.0;
		lang.U[1] = 1.0;
		lang.U[2] = 1.0;
		lang.U[3] = 1.0;
		for (double t = 0.0; t < maxT; t += eps)
		{
			xs.push_back(t);
			ys.push_back(avgPlaquette(lang.U));
			lang.evolveStepImproved(beta, eps);
		}
		plot.plotData(xs, ys, fmt::format("eps = {}", eps));
	}

	Grid_finalize();
}
