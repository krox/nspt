#ifndef NSPT_ALGEBRA_OBSERVABLES
#define NSPT_ALGEBRA_OBSERVABLES

#include "Grid/Grid.h"
#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"
#include "util/gnuplot.h"
#include "util/vector2d.h"

// ugh...
using namespace util;
using namespace Grid;
using namespace Grid::pQCD;

/** A bunch of order-by-order observables measured on the gauge algebra config.
 *  Not very useful for physical insight, but nice to track during simulation */
class AlgebraObservables
{
	using Field = Grid::pQCD::LatticeColourMatrixSeries;

  public:
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

	void measure(const std::array<Field, 4> &A)
	{
		// lattice volume
		double V = A[0]._grid->gSites();

		// trace/herm/norm of field
		{
			RealSeries tmp1 = 0.0;
			RealSeries tmp2 = 0.0;
			RealSeries tmp3 = 0.0;

			for (int mu = 0; mu < 4; ++mu)
			{
				tmp1 += (1.0 / V) * norm2_series(trace(A[mu]));
				tmp2 += (1.0 / V) * norm2_series(Field(A[mu] + adj(A[mu])));
				tmp3 += norm2_series(A[mu]) * (1.0 / V);
			}

			traceA.push_back(asSpan(tmp1));
			hermA.push_back(asSpan(tmp2));
			normA.push_back(asSpan(tmp3));
		}

		// norm of average field (i.e. zero mode)
		avgAx.push_back(asSpan((1.0 / V) * norm2_series(sum(A[0]))));
		avgAy.push_back(asSpan((1.0 / V) * norm2_series(sum(A[1]))));
		avgAz.push_back(asSpan((1.0 / V) * norm2_series(sum(A[2]))));
		avgAt.push_back(asSpan((1.0 / V) * norm2_series(sum(A[3]))));

		// gauge condition
		LatticeColourMatrixSeries w = A[0] - Cshift(A[0], 0, -1);
		w += A[1] - Cshift(A[1], 1, -1);
		w += A[2] - Cshift(A[2], 2, -1);
		w += A[3] - Cshift(A[3], 3, -1);
		gaugeCond.push_back(asSpan((1.0 / V) * norm2_series(w)));
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

#endif
