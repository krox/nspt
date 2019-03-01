#include "Grid/Grid.h"

#include "util/json.hpp"

#include "util/gnuplot.h"
#include "util/stopwatch.h"
#include "util/vector2d.h"
#include <fmt/format.h>

using namespace Grid;
using namespace Grid::QCD;
using namespace util;

#include "nspt/series.h"
#include "nspt/wilson.h"

#include "util/CLI11.hpp"

#include "util/hdf5.h"

using namespace nlohmann;

#include <experimental/filesystem>

/** Perturbative Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	int order;
	GridCartesian *grid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// Fields are expanded in beta^-0.5
	using Field = Series<LatticeColourMatrix>;

	// gauge config
	std::array<Field, 4> U;

	Langevin(std::vector<int> latt, int order, int seed)
	    : order(order),
	      grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      pRNG(grid)
	{
		for (int i = 0; i < order; ++i)
			for (int mu = 0; mu < 4; ++mu)
			{
				U[mu].append(grid);
				U[mu][i] = (i == 0 ? 1.0 : 0.0); // start from unit config
			}
		std::vector<int> pseeds({seed + 0, seed + 1, seed + 2, seed + 3});
		std::vector<int> sseeds({seed + 4, seed + 5, seed + 6, seed + 7});
		pRNG.SeedFixedIntegers(pseeds);
		sRNG.SeedFixedIntegers(sseeds);
	}

	void makeNoise(LatticeColourMatrix &out, double eps)
	{
		/** NOTE: only precise for abelian groups or small-eps limit */
		SU3::GaussianFundamentalLieAlgebraMatrix(pRNG, out, eps);
	}

	void evolveStep(double eps)
	{
		std::array<Field, 4> force;
		for (int i = 0; i < order; ++i)
			for (int mu = 0; mu < 4; ++mu)
				force[mu].append(grid);

		// build force term: F = -eps*D(S) + sqrt(eps) * beta^-1/2 * eta

		for (int mu = 0; mu < 4; ++mu)
		{
			// NOTE: this seems to be the performance bottleneck. Not the
			//       "exp(F)" below
			wilsonDeriv(force[mu], U, mu);
			force[mu] *= -eps;

			LatticeColourMatrix drift(grid);
			makeNoise(drift, std::sqrt(2.0 * eps));
			force[mu][1] += drift;

			// assert(force[mu][0] == 0);
		}

		// evolve U = exp(F) U
		for (int mu = 0; mu < 4; ++mu)
			U[mu] = exp(force[mu]) * U[mu];

		// TODO: check/project unitarity (order by order)
	}

	void landauStep(double alpha)
	{
		Field R;
		for (int i = 0; i < order; ++i)
			R.append(grid);
		R = 0.0;
		for (int mu = 0; mu < 4; ++mu)
		{
			Field Amu = log(U[mu]);
			R += Amu;
			R -= Cshift(Amu, mu, -1); // NOTE: adj(A) = -A
		}

		/** NOTE: without this projection, the non-anti-hermitian part of A will
		 * grow exponentially in time */
		R = Ta(R);

		R = exp(R * (-alpha));

		for (int mu = 0; mu < 4; ++mu)
			U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
	}

	/** compute the algebra elements A=log(U) */
	std::array<Field, 4> algebra()
	{
		std::array<Field, 4> A;
		for (int mu = 0; mu < 4; ++mu)
			A[mu] = log(U[mu]);
		return A;
	}

	void zmreg(bool reunit)
	{
		Field A;
		for (int mu = 0; mu < 4; ++mu)
		{
			A = log(U[mu]);
			for (int i = 0; i < order; ++i)
			{
				ColourMatrix avg = sum(A[i]) * (1.0 / A[i]._grid->gSites());

				// A[i] = A[i] - avg; // This doesn't compile (FIXME)
				std::vector<ColourMatrix> tmp;
				unvectorizeToLexOrdArray(tmp, A[i]);
				for (auto &x : tmp)
					x -= avg;
				vectorizeFromLexOrdArray(tmp, A[i]);

				if (reunit)
					A[i] = Ta(A[i]);
			}
			U[mu] = exp(A);
		}
	}
};

class AlgebraObservables
{
  public:
	/** all these observables are computed sepraately at every order */

	// these are zero except for rounding errors (can be corrected by "reunit")
	vector2d<double> traceA; // avg_µx |Tr(A)|^2
	vector2d<double> hermA;  // avg_µx |A+A^+|^2

	// these drift away from zero as a random walk (can be fixed by "zmreg")
	vector2d<double> avgAx, avgAy, avgAz, avgAt; // |avg_x A(µ)|^2

	// this is zero in Landau gauge, but drifts away in simulation.
	// can be kept moderate by "gaugefix" (no exact fixing needed)
	vector2d<double> gaugeCond; // avg_x |∂µ A(µ)|^2

	// no specific behaviour, but unbounded growth is a bad sign
	vector2d<double> normA; // avg_µx |A|^2

	void measure(const std::array<Langevin::Field, 4> &A)
	{
		int order = A[0].size();
		double V = A[0][0]._grid->gSites();

		{
			std::vector<double> tmp1(order, 0);
			std::vector<double> tmp2(order, 0);
			std::vector<double> tmp3(order, 0);
			for (int i = 0; i < order; ++i)
				for (int mu = 0; mu < 4; ++mu)
				{
					tmp1[i] += (1.0 / V) * norm2(trace(A[mu][i]));
					tmp2[i] +=
					    (1.0 / V) *
					    norm2(LatticeColourMatrix(A[mu][i] + adj(A[mu][i])));
					tmp3[i] += norm2(A[mu][i]) * (1.0 / V);
				}
			traceA.push_back(tmp1);
			hermA.push_back(tmp2);
			normA.push_back(tmp3);
		}

		{
			std::vector<double> tmp1(order, 0);
			std::vector<double> tmp2(order, 0);
			std::vector<double> tmp3(order, 0);
			std::vector<double> tmp4(order, 0);
			for (int i = 0; i < order; ++i)
			{
				tmp1[i] = norm2((1.0 / V) * sum(A[0][i]));
				tmp2[i] = norm2((1.0 / V) * sum(A[1][i]));
				tmp3[i] = norm2((1.0 / V) * sum(A[2][i]));
				tmp4[i] = norm2((1.0 / V) * sum(A[3][i]));
			}
			avgAx.push_back(tmp1);
			avgAy.push_back(tmp2);
			avgAz.push_back(tmp3);
			avgAt.push_back(tmp4);
		}

		{
			std::vector<double> tmp1(order, 0);

			for (int i = 0; i < order; ++i)
			{
				LatticeColourMatrix w = A[0][i] - Cshift(A[0][i], 0, -1);
				w += A[1][i] - Cshift(A[1][i], 1, -1);
				w += A[2][i] - Cshift(A[2][i], 2, -1);
				w += A[3][i] - Cshift(A[3][i], 3, -1);
				tmp1[i] += (1.0 / V) * norm2(w);
			}
			gaugeCond.push_back(tmp1);
		}
	}

	void plot(const std::vector<double> &ts)
	{
		auto plt = [] { return Gnuplot().style("lines").setLogScaleY(); };

		plt().plotData(ts, traceA, "avg |Tr(A)|^2");
		plt().plotData(ts, hermA, "avg |A+A^+|^2");
		plt().plotData(ts, normA, "avg |A|^2");
		plt().plotData(ts, gaugeCond, "avg |∂µ A(µ)|^2");
		plt().plotData(ts, avgAx, "|avg A(x)|^2");
		plt().plotData(ts, avgAy, "|avg A(y)|^2");
		plt().plotData(ts, avgAz, "|avg A(z)|^2");
		plt().plotData(ts, avgAt, "|avg A(t)|^2");
	}
};

int main(int argc, char **argv)
{
	// parameters
	int order = 5;
	double maxt = 20;
	double eps = 0.05;

	int improvement = 1;
	int gaugefix = 1;
	int zmreg = 1;
	int reunit = 1;

	int doPlot = 0;
	int seed = -1;

	std::string filename;
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--order", order, "number of terms in perturbation series");
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

	// data
	if (seed == -1)
		seed = std::random_device()();
	auto lang = Langevin(geom, order, seed);
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
			Series<double> p = avgPlaquette(lang.U);
			plaq.push_back(p);

			fmt::print("t = {}, plaq = ", t);
			for (int i = 0; i < order; ++i)
				fmt::print(", {}", p[i]);
			fmt::print("\n");

			// trace/hermiticity/etc of algebra
			algObs.measure(lang.algebra());

			swMeasure.stop();
		}
	}

	if (filename != "" && filename.find(".json") != std::string::npos)
	{
		json jsonParams;
		jsonParams["order"] = order;
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
		file.setAttribute("order", order);
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
