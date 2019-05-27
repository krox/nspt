#include "modules/module.h"

#include "modules/langevin.h"
#include "modules/lattice_io.h"
#include "modules/meson_2pt.h"
#include "modules/pert_langevin.h"
#include "modules/random_field.h"
#include "modules/unit_field.h"
#include "modules/util.h"
#include "modules/wilson_flow.h"

using namespace Grid;

std::unique_ptr<Module> createModule(const std::string &id, const json &params)
{
	if (id == "UnitField")
		return std::make_unique<MUnitField>(params);
	else if (id == "RandomField")
		return std::make_unique<MRandomField>(params);
	else if (id == "Langevin")
		return std::make_unique<MLangevin>(params);
	else if (id == "PertLangevin")
		return std::make_unique<MPertLangevin>(params);
	else if (id == "DeleteObject")
		return std::make_unique<MDeleteObject>(params);
	else if (id == "WriteField")
		return std::make_unique<MWriteField>(params);
	else if (id == "ReadField")
		return std::make_unique<MReadField>(params);
	else if (id == "WilsonFlow")
		return std::make_unique<MWilsonFlow>(params);
	else if (id == "Meson2pt")
		return std::make_unique<MMeson2pt>(params);
	else
		throw std::range_error(fmt::format("Module '{}' unknown.", id));
}

GridCartesian *Environment::getGrid(const std::vector<int> &latt)
{
	auto simd = GridDefaultSimd((int)latt.size(), vComplex::Nsimd());
	auto mpi = GridDefaultMpi();
	GridCartesian *&grid = grids_[latt];
	if (grid == nullptr)
		grid = new GridCartesian(latt, simd, mpi);
	return grid;
}

GridRedBlackCartesian *Environment::getGridRB(const std::vector<int> &latt)
{
	GridRedBlackCartesian *&grid = gridsRB_[latt];
	if (grid == nullptr)
		grid = new GridRedBlackCartesian(getGrid(latt));
	return grid;
}
