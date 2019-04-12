#include "modules/unit_field.h"

#include "nspt/pqcd.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace Grid::pQCD;

void MUnitField::run(Environment &env)
{
	// TODO: encapsulate "makeGrid" into a global factory
	Grid::GridCartesian *grid = Grid::QCD::SpaceTimeGrid::makeFourDimGrid(
	    params.grid, Grid::GridDefaultSimd(Nd, Grid::vComplex::Nsimd()),
	    Grid::GridDefaultMpi());

	// create Gauge config
	using F = LatticeColourMatrix;
	std::array<F, 4> &U = env.store.create<std::array<F, 4>>(
	    params.field_out, F(grid), F(grid), F(grid), F(grid));

	// set it to unit
	for (int mu = 0; mu < 4; ++mu)
		U[mu] = 1.0;
}
