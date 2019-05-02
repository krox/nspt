#include "modules/random_field.h"

#include <Grid/Grid.h>

using namespace Grid;

void MRandomField::run(Environment &env)
{
	assert(params.grid.size() == 4);
	// TODO: encapsulate "makeGrid" into a global factory
	GridCartesian *grid = QCD::SpaceTimeGrid::makeFourDimGrid(
	    params.grid, Grid::GridDefaultSimd(4, vComplex::Nsimd()),
	    GridDefaultMpi());

	// create Gauge config
	QCD::LatticeGaugeField &U =
	    env.store.create<QCD::LatticeGaugeField>(params.field_out, grid);

	// set it to random SU(3) matrix
	// (actually, "HotConfig()" does not produce fully uniform SU(3) matrices)
	GridParallelRNG pRNG(grid);
	std::vector<int> ps{params.seed};
	pRNG.SeedFixedIntegers(ps);
	QCD::SU3::HotConfiguration(pRNG, U);
}
