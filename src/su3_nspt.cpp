#include "Grid/Grid.h"

#include "util/gnuplot.h"
#include "util/stopwatch.h"
#include <fmt/format.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;
using namespace util;

#include "nspt/series.h"
#include "nspt/wilson.h"

#include "util/CLI11.hpp"

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
};

int main(int argc, char **argv)
{
	// parameters
	int order = 5;
	double maxt = 20;
	double eps = 0.05;
	int gaugefix = 1;
	bool doPlot = false;
	std::string dummy; // ignore options that go directly to grid
	CLI::App app{"NSPT for SU(3) lattice gauge theory"};
	app.add_option("--grid", dummy, "lattice size (e.g. '8.8.8.8')");
	app.add_option("--order", order, "number of terms in perturbation series");
	app.add_option("--maxt", maxt, "Langevin time to integrate");
	app.add_option("--eps", eps, "stepsize for integration");
	app.add_option("--gaugefix", gaugefix, "do stochastic gauge fixing");
	app.add_option("--threads", dummy);
	app.add_flag("--plot", doPlot, "show a plot (requires Gnuplot)");
	CLI11_PARSE(app, argc, argv);

	Grid_init(&argc, &argv);

	// data
	auto lang = Langevin(GridDefaultLatt(), order);

	// plotting
	std::vector<double> xs;
	std::vector<std::vector<double>> ys(order);

	// performance measure
	Stopwatch swEvolve, swMeasure, swLandau;

	// evolve it some time
	for (double t = 0.0; t < maxt; t += eps)
	{
		swMeasure.start();
		Series<double> p = avgPlaquette(lang.U);
		swMeasure.stop();

		fmt::print("t = {}", t);
		xs.push_back(t);
		for (int i = 0; i < order; ++i)
		{
			ys[i].push_back(p[i]);
			fmt::print(", {}", p[i]);
		}
		fmt::print("\n");

		swEvolve.start();
		lang.evolveStep(eps);
		swEvolve.stop();

		// NOTE: the precise amount of gauge-fixing is somewhat arbitrary, but
		//       scaling it similar to the action term seems reasonable
		if (gaugefix)
		{
			swLandau.start();
			lang.landauStep(eps);
			swLandau.stop();
		}
	}

	fmt::print("time for Langevin evolution: {}\n", swEvolve.secs());
	fmt::print("time for Landau gaugefix: {}\n", swLandau.secs());
	fmt::print("time for measurments: {}\n", swMeasure.secs());

	if (doPlot)
	{
		auto plot = Gnuplot();
		plot.style = "lines";
		for (int i = 0; i < order; ++i)
		{
			plot.plotData(xs, ys[i], fmt::format("beta**{}", -0.5 * i));
			double avg = mean(span<const double>(ys[i]).subspan(
			    ys[i].size() / 2, ys[i].size()));
			plot.hline(avg);
		}

		plot.savefig("nstp.pdf");
	}

	Grid_finalize();
}
