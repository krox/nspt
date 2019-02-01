#include "Grid/Grid.h"

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace util;

GridCartesian *grid;
GridRedBlackCartesian *rbGrid;

void thermalize(LatticeGaugeField &U, RealD beta, int count,
                GridSerialRNG &sRNG, GridParallelRNG &pRNG)
{
	LatticeColourMatrix link(grid);
	LatticeColourMatrix staple(grid);

	int subsets[2] = {Even, Odd};
	LatticeInteger one(rbGrid);
	one = 1;
	LatticeInteger mask(grid);

	for (int sweep = 0; sweep < count; sweep++)
	{
		for (int cb = 0; cb < 2; cb++)
		{
			one.checkerboard = subsets[cb];
			mask = zero;
			setCheckerboard(mask, one);

			for (int mu = 0; mu < Nd; mu++)
			{
				ColourWilsonLoops::Staple(staple, U, mu);
				link = peekLorentz(U, mu);

				for (int sub = 0; sub < SU3::su2subgroups(); sub++)
					SU3::SubGroupHeatBath(sRNG, pRNG, beta, link, staple, sub,
					                      5, mask);

				pokeLorentz(U, link, mu);
			}
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
	double betaMin = 0.0;
	double betaMax = 8.0;

	std::vector<double> xs, ys;

	for (int i = 0; i < count; ++i)
	{
		double beta = betaMin + (betaMax - betaMin) / (count - 1) * i;
		thermalize(U, beta, 20, sRNG, pRNG);

		RealD plaq = ColourWilsonLoops::avgPlaquette(U);
		fmt::print("beta = {}, plaq = {}\n", beta, plaq);

		xs.push_back(beta);
		ys.push_back(plaq);
	}
	Gnuplot().plotData(xs, ys);

	Grid_finalize();
}
