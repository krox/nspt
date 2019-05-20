#include "modules/wilson_flow.h"

#include "nspt/grid_utils.h"
#include "nspt/integrator.h"
#include "nspt/pqcd.h"
#include "util/gnuplot.h"
#include <Grid/Grid.h>

using namespace util;
using namespace Grid;

void MWilsonFlow::run(Environment &env)
{
	// get the field and init flow objects
	const QCD::LatticeGaugeField &U =
	    env.store.get<QCD::LatticeGaugeField>(params.field);
	QCD::LatticeGaugeField Usmear = U;

	std::vector<double> ts, t0s, qtop, ps;

	QCD::WilsonFlow<QCD::PeriodicGimplR> WF(1, params.t_step, 1);
	auto SG = QCD::WilsonGaugeAction<QCD::PeriodicGimplR>(3.0);
	for (double t = params.t_step; t <= params.t_max; t += params.t_step)
	{
		WF.smear(Usmear, Usmear);

		// double t0 = WF.energyDensityPlaquette(Usmear);
		double t0 = 2.0 * t * t * SG.S(Usmear) / U._grid->gSites();
		double q = QCD::ColourWilsonLoops::TopologicalCharge(Usmear);
		double p = QCD::ColourWilsonLoops::avgPlaquette(Usmear);
		ts.push_back(t);
		t0s.push_back(t0);
		qtop.push_back(q);
		ps.push_back(p);
		if (primaryTask())
			fmt::print("t = {}, t0 = {}, plaq = {}, qtop = {}\n", t, t0, p, q);
	}
	if (params.plot && primaryTask())
	{
		Gnuplot().plotData(ts, t0s, "t^2 E");
		Gnuplot().plotData(ts, qtop, "Q_top");
		Gnuplot().plotData(ts, ps, "plaq");
	}
}
