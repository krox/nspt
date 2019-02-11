#include "Grid/Grid.h"

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

#include "nspt/wilson.h"

using namespace util;

GridCartesian *grid;
GridRedBlackCartesian *rbGrid;

void thermalize(std::array<LatticeColourMatrix, 4> &U, RealD beta, int count,
                GridSerialRNG &sRNG, GridParallelRNG &pRNG)
{
	LatticeColourMatrix staple(grid);

	int subsets[2] = {Even, Odd};
	LatticeInteger one(rbGrid);
	one = 1;
	LatticeInteger mask(grid);

	for (int sweep = 0; sweep < count; sweep++)
		for (int cb = 0; cb < 2; cb++)
		{
			one.checkerboard = subsets[cb];
			mask = zero;
			setCheckerboard(mask, one);

			for (int mu = 0; mu < Nd; mu++)
			{
				stapleSum(staple, U, mu);
				for (int sub = 0; sub < SU3::su2subgroups(); sub++)
					SU3::SubGroupHeatBath(sRNG, pRNG, beta, U[mu], staple, sub,
					                      5, mask);
			}
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
	std::array<LatticeColourMatrix, 4> U{
	    LatticeColourMatrix(grid), LatticeColourMatrix(grid),
	    LatticeColourMatrix(grid), LatticeColourMatrix(grid)};
	U[0] = 1.0;
	U[1] = 1.0;
	U[2] = 1.0;
	U[3] = 1.0;

	int count = 51;
	double betaMin = 0.0;
	double betaMax = 8.0;

	std::vector<double> xs, ys;

	for (int i = 0; i < count; ++i)
	{
		double beta = betaMin + (betaMax - betaMin) / (count - 1) * i;
		thermalize(U, beta, 20, sRNG, pRNG);

		double plaq = avgPlaquette(U);
		double err = 0.0;
		for (int mu = 0; mu < 4; ++mu)
			err += norm2(LatticeColourMatrix(U[mu] * adj(U[mu]) - 1.0));
		fmt::print("beta = {}, plaq = {}, unit-error = {}\n", beta, plaq, err);

		xs.push_back(beta);
		ys.push_back(plaq);
	}

	Gnuplot().plotData(xs, ys);

	Grid_finalize();
}
