#ifndef NSPT_ALGEBRA_OBSERVABLES
#define NSPT_ALGEBRA_OBSERVABLES

#include "Grid/Grid.h"
#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"
#include "util/gnuplot.h"
#include "util/vector2d.h"

using namespace util;
using namespace Grid;

/** A bunch of order-by-order observables measured on the gauge algebra config.
 *  Not very useful for physical insight, but nice to track during simulation */
class AlgebraObservables
{
	using Field = pQCD::LatticeColourMatrixSeries;

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

	void measure(const std::array<Field, 4> &A);

	void plot(const std::vector<double> &ts)
	{
		auto plt = [] {
			auto p = Gnuplot();
			p.style("lines").setLogScaleY();
			return p;
		};

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
