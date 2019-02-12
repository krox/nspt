#include "Grid/Grid.h"

#include "util/gnuplot.h"
#include <fmt/format.h>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;
using namespace util;

#include "nspt/series.h"
#include "nspt/wilson.h"

/** Perturbative Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	int degree;
	GridCartesian *grid;
	GridRedBlackCartesian *rbGrid;
	GridParallelRNG pRNG;
	GridSerialRNG sRNG;

	// Fields are expanded in beta^-0.5
	using Field = Series<LatticeColourMatrix>;

	// gauge config
	std::array<Field, 4> U;

	Langevin(std::vector<int> latt, int degree)
	    : degree(degree),
	      grid(SpaceTimeGrid::makeFourDimGrid(
	          latt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi())),
	      rbGrid(SpaceTimeGrid::makeFourDimRedBlackGrid(grid)), pRNG(grid)
	{
		for (int i = 0; i < degree; ++i)
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
		for (int i = 0; i < degree; ++i)
			for (int mu = 0; mu < 4; ++mu)
				force[mu].append(grid);

		// build force term: F = -eps*D(S) + sqrt(eps) * beta^-1/2 * eta
		for (int mu = 0; mu < 4; ++mu)
		{
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
};

int main(int argc, char **argv)
{
	Grid_init(&argc, &argv);

	// parameters
	int degree = 5;
	double maxT = 10;
	double eps = 0.05;

	// data
	auto lang = Langevin(GridDefaultLatt(), degree);
	std::vector<double> xs;
	std::vector<std::vector<double>> ys(degree);

	// evolve it some time
	for (double t = 0.0; t < maxT; t += eps)
	{
		Series<double> p = avgPlaquette(lang.U);

		fmt::print("t = {}", t);
		xs.push_back(t);
		for (int i = 0; i < degree; ++i)
		{
			ys[i].push_back(p[i]);
			fmt::print(", {}", p[i]);
		}
		fmt::print("\n");

		lang.evolveStep(eps);
	}

	auto plot = Gnuplot();
	plot.style = "lines";
	for (int i = 0; i < degree; ++i)
	{
		plot.plotData(xs, ys[i], fmt::format("beta**{}", -0.5 * i));
		double avg = mean(
		    span<const double>(ys[i]).subspan(ys[i].size() / 2, ys[i].size()));
		plot.hline(avg);
	}

	plot.savefig("nstp.pdf");

	Grid_finalize();
}
