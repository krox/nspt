#include "Grid/Grid.h"

#include "util/json.hpp"

#include "util/CLI11.hpp"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/stopwatch.h"
#include "util/vector2d.h"
#include <experimental/filesystem>
#include <fmt/format.h>

#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"

using namespace Grid;
using Grid::QCD::SpaceTimeGrid;
using namespace Grid::pQCD;
using namespace util;

#include "nspt/algebra_observables.h"
#include "nspt/wilson.h"

using namespace nlohmann;

/** Perturbative Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	GridCartesian *grid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// Fields are expanded in beta^-0.5
	using Field = LatticeColourMatrixSeries;

	// gauge config
	std::array<Field, 4> U;

	Langevin(std::vector<int> latt, int seed)
	    : grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      pRNG(grid), U{Field(grid), Field(grid), Field(grid), Field(grid)}
	{
		// start from unit config
		for (int mu = 0; mu < 4; ++mu)
			U[mu] = 1.0;

		std::vector<int> pseeds({seed + 0, seed + 1, seed + 2, seed + 3});
		std::vector<int> sseeds({seed + 4, seed + 5, seed + 6, seed + 7});
		pRNG.SeedFixedIntegers(pseeds);
		sRNG.SeedFixedIntegers(sseeds);
	}

	void makeNoise(LatticeColourMatrix &out, double eps)
	{
		gaussian(pRNG, out);
		out *= eps * std::sqrt(0.5); // normalization of SU(3) generators
		out = Ta(out);
	}

	void evolveStep(double eps)
	{
		std::array<Field, 4> force{Field(grid), Field(grid), Field(grid),
		                           Field(grid)};

		// build force term: F = -eps*D(S) + sqrt(eps) * beta^-1/2 * eta

		for (int mu = 0; mu < 4; ++mu)
		{
			// NOTE: this seems to be the performance bottleneck. Not the
			//       "exp(F)" below
			wilsonDeriv(force[mu], U, mu);
			force[mu] *= -eps;

			LatticeColourMatrix drift(grid);
			makeNoise(drift, std::sqrt(2.0 * eps));

			// noise enters the force at order beta^-1/2
			LatticeColourMatrix tmp = peekSeries(force[mu], 1);
			tmp += drift;
			pokeSeries(force[mu], tmp, 1);
		}

		// evolve U = exp(F) U
		for (int mu = 0; mu < 4; ++mu)
			U[mu] = expMat(force[mu], 1.0, No - 1) * U[mu];
	}

	void landauStep(double alpha)
	{
		Field R(grid);

		R = 0.0;
		for (int mu = 0; mu < 4; ++mu)
		{
			// turns out, both of these work fine to suppress the drift in
			// Langevin time. Though Ta() is obviously faster
			Field Amu = logMat(U[mu]);
			// Field Amu = Ta(U[mu]);

			R += Amu;
			R -= Cshift(Amu, mu, -1); // NOTE: adj(A) = -A
		}

		/** NOTE: without this projection, the non-anti-hermitian part of A will
		 * grow exponentially in time */
		R = Ta(R);

		R = expMat(R, -alpha, No - 1);

		for (int mu = 0; mu < 4; ++mu)
			U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
	}

	/** compute the algebra elements A=log(U) */
	std::array<Field, 4> algebra()
	{
		std::array<Field, 4> A{Field(grid), Field(grid), Field(grid),
		                       Field(grid)};
		for (int mu = 0; mu < 4; ++mu)
			A[mu] = logMat(U[mu]);
		return A;
	}

	void zmreg(bool reunit)
	{
		Field A{grid};
		for (int mu = 0; mu < 4; ++mu)
		{
			A = logMat(U[mu]);
			ColourMatrixSeries avg = sum(A) * (1.0 / A._grid->gSites());

			// A[i] = A[i] - avg; // This doesn't compile (FIXME)
			std::vector<ColourMatrixSeries> tmp;
			unvectorizeToLexOrdArray(tmp, A);
			for (auto &x : tmp)
				x -= avg;
			vectorizeFromLexOrdArray(tmp, A);

			if (reunit)
				A = Ta(A);

			U[mu] = expMat(A, 1.0, No - 1);
		}
	}
};

int main(int argc, char **argv)
{
	// parameters
	double maxt = 20;
	double eps = 0.05;

	int improvement = 1;
	int gaugefix = 1;
	int zmreg = 1;
	int reunit = 1;

	int doPlot = 0;
	int seed = -1;
	int verbosity = 1;

	std::string filename;
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--maxt", maxt, "Langevin time to integrate");
	app.add_option("--eps", eps, "stepsize for integration");
	app.add_option("--improvement", improvement, "use improved integrator");
	app.add_option("--gaugefix", gaugefix, "do stochastic gauge fixing");
	app.add_option("--zmreg", zmreg, "explicitly remove zero modes");
	app.add_option("--reunit", reunit, "explicitly project onto group/algebra");
	app.add_option("--threads", dummy);
	app.add_option("--plot", doPlot, "show a plot (requires Gnuplot)");
	app.add_option("--filename", filename, "output file (json format)");
	app.add_option("--seed", seed, "seed for rng (default = unpredictable)");
	app.add_option("--verbosity", verbosity, "verbosity (default = 1)");
	CLI11_PARSE(app, argc, argv);

	if (filename != "" && std::experimental::filesystem::exists(filename))
	{
		fmt::print("{} already exists. skipping.\n", filename);
		return 0;
	}

	if (filename != "")
		if (filename.find(".json") == std::string::npos &&
		    filename.find(".h5") == std::string::npos)
		{
			fmt::print("ERROR: unrecognized file ending: {}\n", filename);
			return -1;
		}

	Grid_init(&argc, &argv);
	std::vector<int> geom = GridDefaultLatt();

	fmt::print("L = {}, eps = {}, maxt = {}\n", geom[0], eps, maxt);

	// data
	if (seed == -1)
		seed = std::random_device()();
	auto lang = Langevin(geom, seed);
	std::vector<double> ts;
	vector2d<double> plaq;
	AlgebraObservables algObs;

	// performance measure
	Stopwatch swEvolve, swMeasure, swLandau, swZmreg;

	// evolve it some time
	for (double t = 0.0; t < maxt; t += eps)
	{
		// step 1: langevin evolution
		swEvolve.start();
		if (improvement == 0)
			lang.evolveStep(eps);
		else
			assert(false);
		swEvolve.stop();

		// step 2: stochastic gauge-fixing and zero-mode regularization
		// NOTE: the precise amount of gauge-fixing is somewhat arbitrary, but
		//       don't scale it with epsilon.

		swLandau.start();
		for (int i = 0; i < gaugefix; ++i)
			lang.landauStep(0.1);
		swLandau.stop();

		if (zmreg)
		{
			swZmreg.start();
			lang.zmreg(reunit);
			swZmreg.stop();
		}

		// step 3: measurements
		{
			swMeasure.start();
			ts.push_back(t + eps);

			// plaquette
			RealSeries p = avgPlaquette(lang.U);
			plaq.push_back(asSpan(p));

			if (verbosity >= 1)
				fmt::print("t = {}, plaq = {:.5}\n", t, p);

			// trace/hermiticity/etc of algebra
			algObs.measure(lang.algebra());

			swMeasure.stop();
		}
	}

	if (filename != "")
		fmt::print("writing results to '{}'\n", filename);

	if (filename != "" && filename.find(".json") != std::string::npos)
	{
		json jsonParams;
		jsonParams["order"] = No;
		jsonParams["geom"] = geom;
		jsonParams["maxt"] = maxt;
		jsonParams["eps"] = eps;
		jsonParams["improvement"] = improvement;
		jsonParams["gaugefix"] = gaugefix;
		jsonParams["zmreg"] = zmreg;
		jsonParams["reunit"] = reunit;

		json jsonOut;
		jsonOut["params"] = jsonParams;
		jsonOut["ts"] = ts;
		jsonOut["plaq"] = plaq;
		jsonOut["traceA"] = algObs.traceA;
		jsonOut["hermA"] = algObs.hermA;
		jsonOut["normA"] = algObs.normA;
		jsonOut["gaugeCond"] = algObs.gaugeCond;
		jsonOut["avgAx"] = algObs.avgAx;
		jsonOut["avgAy"] = algObs.avgAy;
		jsonOut["avgAz"] = algObs.avgAz;
		jsonOut["avgAt"] = algObs.avgAt;
		std::ofstream(filename) << jsonOut.dump(2) << std::endl;
	}
	else if (filename != "" && filename.find(".h5") != std::string::npos)
	{
		auto file = DataFile::create(filename);
		file.setAttribute("order", No);
		file.setAttribute("geom", geom);
		file.setAttribute("maxt", maxt);
		file.setAttribute("eps", eps);
		file.setAttribute("improvement", improvement);
		file.setAttribute("gaugefix", gaugefix);
		file.setAttribute("zmreg", zmreg);
		file.setAttribute("reunit", reunit);

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

	fmt::print("time for Langevin evolution: {}\n", swEvolve.secs());
	fmt::print("time for Landau gaugefix: {}\n", swLandau.secs());
	fmt::print("time for ZM regularization: {}\n", swZmreg.secs());
	fmt::print("time for measurments: {}\n", swMeasure.secs());

	if (doPlot >= 1)
		Gnuplot().style("lines").plotData(ts, plaq, "plaq");
	if (doPlot >= 2)
		algObs.plot(ts);

	Grid_finalize();
}
