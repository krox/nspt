#include "modules/lattice_io.h"

#include "nspt/lattice_io.h"
#include <Grid/Grid.h>

using namespace Grid;

void MWriteField::run(Environment &env)
{
	auto &U = env.store.get<QCD::LatticeGaugeField>(params.field);
	if (params.format == "nersc")
	{
		// no idea what these are intended for. But looking at
		// github.com/paboyle/Grid/blob/develop/Grid/parallelIO/NerscIO.h
		// they seem to be unused
		int two_row = 0;
		int bits32 = 0;

		QCD::NerscIO::writeConfiguration(U, params.filename, two_row, bits32);
	}
	else
	{
		writeConfig(params.filename, U);
	}
}

void MReadField::run(Environment &env)
{
	if (params.format == "openqcd")
	{
		std::vector<int> grid = getGridFromFileOpenQCD(params.filename);
		auto &U = env.store.create<QCD::LatticeGaugeField>(params.field,
		                                                   env.getGrid(grid));
		readConfigOpenQCD(params.filename, U);
	}
	else if (params.format == "nersc")
	{
		auto grid = env.getGrid(params.grid);
		auto &U = env.store.create<QCD::LatticeGaugeField>(params.field, grid);
		FieldMetaData header;
		QCD::NerscIO::readConfiguration(U, header, params.filename);
	}
	else
	{
		std::vector<int> grid = getGridFromFile(params.filename);
		auto &U = env.store.create<QCD::LatticeGaugeField>(params.field,
		                                                   env.getGrid(grid));
		readConfig(params.filename, U);
	}
}
