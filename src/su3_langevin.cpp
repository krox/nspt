#include "Grid/Grid.h"

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace util;

GridCartesian *grid;
GridRedBlackCartesian *rbGrid;

void evolve(LatticeGaugeField &U, RealD beta, double eps, int count,
            [[maybe_unused]] GridSerialRNG &sRNG, GridParallelRNG &pRNG)
{
	LatticeGaugeField force(grid);
	LatticeColourMatrix link(grid);
	LatticeColourMatrix drift(grid);
	LatticeColourMatrix rot(grid);

	WilsonGaugeActionD action(beta);

	for (int i = 0; i < count; ++i)
	{
		// force term from action
		action.deriv(U, force);

		for (int mu = 0; mu < Nd; ++mu)
		{
			SU3::GaussianFundamentalLieAlgebraMatrix(
			    pRNG, drift, std::sqrt(2.0) * std::sqrt(eps));
			// drift = Ta(drift);
			drift += -eps * peekLorentz(force, mu);
			SU3::taExp(drift, rot);
			link = peekLorentz(U, mu);
			pokeLorentz(U, rot * link, mu);
		}

		ProjectOnGroup(U);
	}
}

int main(int argc, char **argv)
{
	Grid_init(&argc, &argv);

	// use default grid size (i.e. from command line)
	grid = SpaceTimeGrid::makeFourDimGrid(
	    GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()),
	    GridDefaultMpi());
	rbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(grid);

	// arbitrary random seeds
	std::vector<int> pseeds({1, 2, 3, 4, 5});
	std::vector<int> sseeds({6, 7, 8, 9, 10});
	GridParallelRNG pRNG(grid);
	pRNG.SeedFixedIntegers(pseeds);
	GridSerialRNG sRNG;
	sRNG.SeedFixedIntegers(sseeds);

	// gauge config (hot start at small beta)
	LatticeGaugeField U(grid);
	SU3::HotConfiguration(pRNG, U);

	int count = 51;
	double betaMin = 8.0;
	double betaMax = 0.0;

	std::vector<double> xs, ys;

	for (int i = 0; i < count; ++i)
	{
		double beta = betaMin + (betaMax - betaMin) / (count - 1) * i;
		evolve(U, beta, 0.02, 500, sRNG, pRNG);

		RealD plaq = ColourWilsonLoops::avgPlaquette(U);
		fmt::print("beta = {}, plaq = {}\n", beta, plaq);

		xs.push_back(beta);
		ys.push_back(plaq);
	}
	Gnuplot().plotData(xs, ys);

	Grid_finalize();
}
