#include "modules/gaugefix.h"

#include "nspt/grid_utils.h"
#include "nspt/integrator.h"
#include "nspt/pqcd.h"
#include "util/gnuplot.h"
#include <Grid/Grid.h>

using namespace util;
using namespace Grid;

GaugefixIntegrator::GaugefixIntegrator(Grid::GridCartesian *grid)
    : R(grid), tmp(grid), prec(grid), fft(grid)
{
	assert(grid);
	// Idea of Fourier transform: use Laplace-operator as preconditioner.
	// Lattice laplace has known eigenvalues p^2 = 4 sum_mu sin^2(n pi / L)
	prec = 0.0;
	for (int mu = 0; mu < QCD::Nd; ++mu)
	{
		LatticeReal co(grid);
		LatticeCoordinate(co, mu);
		double L = grid->FullDimensions()[mu];
		prec += 4.0 * sin(co * (M_PI / L)) * sin(co * (M_PI / L));
	}
	RealD p2max = 4.0 * QCD::Nd; // = max(prec)

	LatticeReal def(grid);

	// set zero modes to one (yes, pretty arbitrary)
	def = 1.0;
	prec = where(prec == 0.0, def, prec);
	def = p2max;

	prec = def / prec;
}

double GaugefixIntegrator::gaugecond(const GaugeField &U) const
{
	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		tmp = Ta(QCD::peekLorentz(U, mu));
		R += tmp;
		R -= Cshift(tmp, mu, -1); // NOTE: adj(A) = -A
	}

	return norm2(R) / U._grid->gSites();
}

void GaugefixIntegrator::step(GaugeField &U, double eps)
{
	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		tmp = Ta(QCD::peekLorentz(U, mu));
		R += tmp;
		R -= Cshift(tmp, mu, -1);
	}

	// Fourier acceleration
	fft.FFT_all_dim(tmp, R, -1); // forward Fourier (normalization 1)
	tmp *= toComplex(prec);      // why is the "toComplex" necessary here?
	fft.FFT_all_dim(R, tmp, +1); // backward Fourier (normalization 1/V)

	R *= -toComplex(eps);
	R = expMat(R, 1.0);
	for (int mu = 0; mu < 4; ++mu)
	{
		tmp = Cshift(adj(R), mu, 1);
		parallel_for(int ss = 0; ss < U._grid->oSites(); ss++)
		{
			U._odata[ss](mu) =
			    R._odata[ss]() * U._odata[ss](mu) * tmp._odata[ss]();
		}
	}
}

void MGaugefix::run(Environment &env)
{
	// get the field and init flow objects
	QCD::LatticeGaugeField &U =
	    env.store.get<QCD::LatticeGaugeField>(params.field);

	auto gf = GaugefixIntegrator(dynamic_cast<GridCartesian *>(U._grid));

	for (int iter = 0; iter < params.iter_max; ++iter)
	{
		gf.step(U, 0.0625);

		double gcond = gf.gaugecond(U);
		fmt::print("iter={}, gcond={}\n", iter + 1, gcond);
		if (gcond <= params.gcond)
			return;
	}
}
