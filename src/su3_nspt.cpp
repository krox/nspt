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

using namespace nlohmann;

#include <experimental/filesystem>

/** Perturbative Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	int order;
	GridCartesian *grid;
	GridRedBlackCartesian *rbGrid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// Fields are expanded in beta^-0.5
	using Field = Series<LatticeColourMatrix>;

	// gauge config
	std::array<Field, 4> U;

	Langevin(std::vector<int> latt, int order)
	    : order(order),
	      grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      rbGrid(SpaceTimeGrid::makeFourDimRedBlackGrid(grid)), pRNG(grid)
	{
		for (int i = 0; i < order; ++i)
			for (int mu = 0; mu < 4; ++mu)
			{
				U[mu].append(grid);
				U[mu][i] = (i == 0 ? 1.0 : 0.0); // start from unit config
			}
		std::vector<int> pseeds({1, 2, 3, 4, 5});
		std::vector<int> sseeds({6, 7, 8, 9, 10});
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

	void landauStep(double eps)
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

		R = exp(R * (-0.5 * eps));

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

int main(int argc, char **argv)
{
	// parameters
	int order = 5;
	double maxt = 20;
	double eps = 0.05;
	int gaugefix = 1;
	int zmreg = 2;
	bool doPlot = false;
	std::string filename;
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--order", order, "number of terms in perturbation series");
	app.add_option("--maxt", maxt, "Langevin time to integrate");
	app.add_option("--eps", eps, "stepsize for integration");
	app.add_option("--gaugefix", gaugefix, "do stochastic gauge fixing");
	app.add_option("--zmreg", zmreg, "explicitly remove zero modes");
	app.add_option("--threads", dummy);
	app.add_flag("--plot", doPlot, "show a plot (requires Gnuplot)");
	app.add_option("--filename", filename, "output file (json format)");
	CLI11_PARSE(app, argc, argv);

	if (filename != "" && std::experimental::filesystem::exists(filename))
	{
		fmt::print("{} already exists. skipping.\n", filename);
		return 0;
	}

	Grid_init(&argc, &argv);
	std::vector<int> geom = GridDefaultLatt();

	// data
	auto lang = Langevin(geom, order);
	std::vector<double> ts;
	vector2d<double> plaq, traceA, hermA, normA;

	// performance measure
	Stopwatch swEvolve, swMeasure, swLandau, swZmreg;

	// evolve it some time
	for (double t = 0.0; t < maxt; t += eps)
	{
		// step 1: langevin evolution
		swEvolve.start();
		lang.evolveStep(eps);
		swEvolve.stop();

		// step 2: stochastic gauge-fixing and zero-mode regularization
		// NOTE: the precise amount of gauge-fixing is somewhat arbitrary, but
		//       scaling it similar to the action term seems reasonable
		if (gaugefix)
		{
			swLandau.start();
			lang.landauStep(eps);
			swLandau.stop();
		}

		if (zmreg)
		{
			swZmreg.start();
			lang.zmreg(zmreg >= 2);
			swZmreg.stop();
		}

		// step 3: measurements
		{
			swMeasure.start();
			ts.push_back(t + eps);

			// plaquette
			Series<double> p = avgPlaquette(lang.U);
			plaq.push_back(p);

			// trace/hermiticity/norm of algebra
			auto alg = lang.algebra();
			std::vector<double> tA(order, 0.0);
			std::vector<double> hA(order, 0.0);
			std::vector<double> nA(order, 0.0);
			double V = alg[0][0]._grid->gSites();
			for (int i = 0; i < order; ++i)
				for (int mu = 0; mu < 4; ++mu)
				{
					tA[i] += norm2(trace(alg[mu][i])) / V;
					hA[i] += norm2(LatticeColourMatrix(alg[mu][i] +
					                                   adj(alg[mu][i]))) /
					         V;

					nA[i] += norm2(alg[mu][i]) / V;
				}
			traceA.push_back(tA);
			hermA.push_back(hA);
			normA.push_back(nA);

			fmt::print("t = {}, plaq = ", t);
			for (int i = 0; i < order; ++i)
				fmt::print(", {}", p[i]);
			fmt::print("\n");

			swMeasure.stop();
		}
	}

	if (filename != "")
	{
		json jsonParams;
		jsonParams["order"] = order;
		jsonParams["geom"] = geom;
		jsonParams["maxt"] = maxt;
		jsonParams["eps"] = eps;
		jsonParams["gaugefix"] = gaugefix;
		jsonParams["improvement"] = 0;
		jsonParams["zmreg"] = zmreg;

		json jsonOut;
		jsonOut["params"] = jsonParams;
		jsonOut["ts"] = ts;
		jsonOut["plaq"] = plaq;
		jsonOut["traceA"] = traceA;
		jsonOut["hermA"] = hermA;
		jsonOut["normA"] = normA;
		std::ofstream(filename) << jsonOut.dump(2) << std::endl;
	}

	fmt::print("time for Langevin evolution: {}\n", swEvolve.secs());
	fmt::print("time for Landau gaugefix: {}\n", swLandau.secs());
	fmt::print("time for ZM regularization: {}\n", swZmreg.secs());
	fmt::print("time for measurments: {}\n", swMeasure.secs());

	if (doPlot)
	{
		Gnuplot().style("lines").plotData(ts, plaq, "plaq");
		Gnuplot().style("lines").plotData(ts, traceA, "trace(A)");
		Gnuplot().style("lines").plotData(ts, hermA, "herm(A)");
		Gnuplot().style("lines").plotData(ts, normA, "norm(A)");
	}

	Grid_finalize();
}
