#include "modules/random_field.h"

#include "nspt/pqcd.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace Grid::pQCD;

void MRandomField::run(Environment &env)
{
	// TODO: encapsulate "makeGrid" into a global factory
	Grid::GridCartesian *grid = Grid::QCD::SpaceTimeGrid::makeFourDimGrid(
	    params.grid, Grid::GridDefaultSimd(Nd, Grid::vComplex::Nsimd()),
	    Grid::GridDefaultMpi());

	// create Gauge config
	using F = LatticeColourMatrix;
	std::array<F, 4> &U = env.store.create<std::array<F, 4>>(
	    params.field_out, F(grid), F(grid), F(grid), F(grid));

	// set it to random SU(3) matrix
	F drift(grid);
	GridParallelRNG pRNG(grid);
	std::vector<int> ps{params.seed};
	pRNG.SeedFixedIntegers(ps);
	for (int mu = 0; mu < 4; ++mu)
	{
		gaussian(pRNG, drift);
		drift = Ta(drift);
		U[mu] = expMat(drift, 1.0);
		gaussian(pRNG, drift);
		drift = Ta(drift);
		U[mu] = expMat(drift, 1.0) * U[mu];
	}
}
