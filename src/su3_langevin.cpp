#include "Grid/Grid.h"

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace util;

GridCartesian *grid;
GridRedBlackCartesian *rbGrid;

/** Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	GridCartesian *grid;
	GridRedBlackCartesian *rbGrid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// gauge config
	LatticeGaugeField U;

	// temporaries
	LatticeGaugeField force, force2;
	LatticeColourMatrix Umu, drift, rot, tmp1, tmp2;

	/** NOTE: does not initialize gauge field */
	Langevin(std::vector<int> latt)
	    : grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      rbGrid(SpaceTimeGrid::makeFourDimRedBlackGrid(grid)), pRNG(grid),
	      U(grid), force(grid), force2(grid), Umu(grid), drift(grid), rot(grid),
	      tmp1(grid), tmp2(grid)
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
		WilsonGaugeActionD action(beta);
		action.deriv(U, force);

		for (int mu = 0; mu < Nd; ++mu)
		{
			makeNoise(drift, std::sqrt(2.0 * eps));
			drift += -eps * peekLorentz(force, mu);
			SU3::taExp(drift, rot);
			Umu = peekLorentz(U, mu);
			pokeLorentz(U, rot * Umu, mu);
		}

		ProjectOnGroup(U);
	}

	/** second order method (Heun method) */
	void evolveStepImproved(RealD beta, double eps)
	{
		WilsonGaugeActionD action(beta);

		// force at t=0
		action.deriv(U, force);

		// evolve with force + drift to t=eps
		for (int mu = 0; mu < Nd; ++mu)
		{
			makeNoise(drift, std::sqrt(2.0 * eps));
			drift += -eps * peekLorentz(force, mu);
			SU3::taExp(drift, rot);
			Umu = peekLorentz(U, mu);
			pokeLorentz(U, rot * Umu, mu);
		}

		// force at new point
		action.deriv(U, force2);

		// correct to 0.5*(force + force2)
		for (int mu = 0; mu < Nd; ++mu)
		{
			drift =
			    -eps * 0.5 * (peekLorentz(force2, mu) - peekLorentz(force, mu));
			SU3::taExp(drift, rot);
			Umu = peekLorentz(U, mu);
			pokeLorentz(U, rot * Umu, mu);
		}
		ProjectOnGroup(U);
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

		lang.U = 1.0;
		for (double t = 0.0; t < maxT; t += eps)
		{
			xs.push_back(t);
			ys.push_back(ColourWilsonLoops::avgPlaquette(lang.U));
			lang.evolveStepImproved(beta, eps);
		}
		plot.plotData(xs, ys, fmt::format("eps = {}", eps));
	}

	Grid_finalize();
}
