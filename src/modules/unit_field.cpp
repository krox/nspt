#include "modules/unit_field.h"

#include "nspt/pqcd.h"
#include <Grid/Grid.h>

using namespace Grid;

void MUnitField::run(Environment &env)
{
	// get the grid
	assert(params.grid.size() == 4);
	GridCartesian *grid = env.getGrid(params.grid);

	// create Gauge config and set it to unit
	if (params.order == 0)
	{
		QCD::LatticeGaugeField &U =
		    env.store.create<QCD::LatticeGaugeField>(params.field_out, grid);
		U = 1.0;
	}
	else
	{
		assert(params.order == pQCD::No);
		using F = pQCD::LatticeColourMatrixSeries;
		std::array<F, 4> &U = env.store.create<std::array<F, 4>>(
		    params.field_out, F(grid), F(grid), F(grid), F(grid));
		for (int mu = 0; mu < 4; ++mu)
			U[mu] = 1.0;
	}
}
