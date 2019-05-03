#include "modules/random_field.h"

#include <Grid/Grid.h>

using namespace Grid;

void MRandomField::run(Environment &env)
{
	// get the grid
	assert(params.grid.size() == 4);
	GridCartesian *grid = env.getGrid(params.grid);

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
