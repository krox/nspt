#include "modules/pert_langevin.h"

#include "nspt/algebra_observables.h"
#include "nspt/grid_utils.h"
#include "nspt/integrator.h"
#include "nspt/nspt.h"
#include "nspt/pqcd.h"
#include "util/hdf5.h"
#include "util/vector2d.h"
#include <Grid/Grid.h>
#include <array>

using namespace Grid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/wilson.h"

void MPertLangevin::run(Environment &env)
{
	// gauge field
	using Field = LatticeColourMatrixSeries;
	std::array<Field, 4> &U = env.store.get<std::array<Field, 4>>(params.field);
	Grid::GridBase *grid = U[0]._grid;

	// temporaries
	std::array<Field, 4> A = {Field(grid), Field(grid), Field(grid),
	                          Field(grid)};

	// create the numerical integrator
	std::unique_ptr<Integrator> integrator;
	if (params.improvement == 0)
		integrator = std::make_unique<EulerIntegrator>(grid, params.seed);
	else if (params.improvement == 1)
		integrator = std::make_unique<BFIntegrator>(grid, params.seed);
	else if (params.improvement == 2)
		integrator = std::make_unique<BauerIntegrator>(grid, params.seed);
	else
		assert(false);

	std::unique_ptr<LandauIntegrator> landauIntegrator;
	RealSeries landauEps = params.gaugefix_fourier ? 0.0625 : 0.1;
	if (params.gaugefix)
	{
		landauIntegrator = std::make_unique<LandauIntegrator>(
		    dynamic_cast<Grid::GridCartesian *>(grid));
		landauIntegrator->fourierAccel = params.gaugefix_fourier;
	}

	// track some observables during simulation
	std::vector<double> ts;
	vector2d<double> plaq;
	AlgebraObservables algObs;

	// run the Markov process
	for (int i = 0; i < params.count; ++i)
	{
		// integration step
		integrator->step(U, params.eps);

		// gaugefixing
		for (int j = 0; j < params.gaugefix; ++j)
			landauIntegrator->step(U, landauEps);

		// zero-mode regularization
		if (params.zmreg)
			removeZero(U, params.reunit);

		// track some basic observables
		RealSeries p = avgPlaquette(U); // plaquette

		ts.push_back((i + 1) * params.eps);
		plaq.push_back(asSpan(p));

		// trace/hermiticity/etc of algebra
		for (int mu = 0; mu < 4; ++mu)
			A[mu] = logMatFast(U[mu]);
		algObs.measure(A);

		if ((i + 1) % 10 == 0)
			if (primaryTask())
				fmt::print("k = {}/{}, plaq = {:.5}, gcond = {:.5f}\n", i + 1,
				           params.count, p, algObs.gaugeCond.back());
	}

	// write results to hdf5 file
	if (primaryTask() && params.filename != "")
	{
		fmt::print("writing results to '{}'\n", params.filename);

		auto file = DataFile::create(params.filename);
		file.setAttribute("order", No);
		file.setAttribute("geom", grid->FullDimensions());
		file.setAttribute("count", params.count);
		file.setAttribute("eps", params.eps);
		file.setAttribute("improvement", params.improvement);
		file.setAttribute("gaugefix", params.gaugefix);
		file.setAttribute("zmreg", params.zmreg);
		file.setAttribute("reunit", params.reunit);

		file.createData("ts", ts);
		file.createData("plaq", plaq);
		file.createData("traceA", algObs.traceA);
		file.createData("hermA", algObs.hermA);
		file.createData("normA", algObs.normA);
		file.createData("gaugeCond", algObs.gaugeCond);
		file.createData("avgAx", algObs.avgAx);
		file.createData("avgAy", algObs.avgAy);
		file.createData("avgAz", algObs.avgAz);
		file.createData("avgAt", algObs.avgAt);
	}
}
