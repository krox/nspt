#include "modules/lattice_io.h"

#include "nspt/lattice_io.h"
#include <Grid/Grid.h>

using namespace Grid;

void MWriteField::run(Environment &env)
{
	auto &U = env.store.get<QCD::LatticeGaugeField>(params.field);
	writeConfig(params.filename, U);
}

void MReadField::run(Environment &env)
{
	std::vector<int> grid = getGridFromFile(params.filename);
	auto &U = env.store.create<QCD::LatticeGaugeField>(params.field,
	                                                   env.getGrid(grid));
	readConfig(params.filename, U);
}
