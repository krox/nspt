#include "nspt/integrator.h"

#include "nspt/grid_utils.h"
#include <cmath>

using namespace Grid;
using namespace pQCD;

#include "nspt/wilson.h"

/** generate gaussian noise with <eta^2> = 2 */
static void makeNoise(LatticeColourMatrix &out, GridParallelRNG &pRNG)
{
	// TODO: really check that this produces the correct normalization
	gaussian(pRNG, out);
	out = Ta(out);
}

/** compute V = exp(aX +bY<<1))U. aliasing is allowed */
static void evolve(LatticeColourMatrixSeries &V, double a,
                   const LatticeColourMatrixSeries &X, double b,
                   const LatticeColourMatrix &Y,
                   const LatticeColourMatrixSeries &U)
{
	assert(No >= 2);
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vColourMatrixSeries tmp = a * X._odata[ss];
		tmp()()(1) += b * Y._odata[ss]()()();
		V._odata[ss] = ExponentiateFast(tmp, 1.0) * U._odata[ss];
	}
}

/** compute V = exp(aX +bY<<2 + cZ<<4))U. aliasing is allowed */
static void evolve(LatticeColourMatrixSeries &V, double a,
                   const LatticeColourMatrixSeries &X, double b,
                   const LatticeColourMatrix &Y, double c,
                   const LatticeColourMatrixSeries &Z,
                   const LatticeColourMatrixSeries &U)
{
	assert(No >= 2);
	conformable(V._grid, X._grid);
	conformable(V._grid, Y._grid);
	conformable(V._grid, Z._grid);
	conformable(V._grid, U._grid);

	parallel_for(int ss = 0; ss < V._grid->oSites(); ss++)
	{
		vColourMatrixSeries tmp = a * X._odata[ss];
		tmp()()(1) += b * Y._odata[ss]()()();
		for (int i = 2; i < No; ++i)
			tmp()()(i) += c * Z._odata[ss]()()(i - 2);
		V._odata[ss] = ExponentiateFast(tmp, 1.0) * U._odata[ss];
	}
}

EulerIntegrator::EulerIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{grid}, force{Field(grid), Field(grid), Field(grid),
                                     Field(grid)}
{
	pRNG.SeedFixedIntegers({seed});
}

void EulerIntegrator::step(std::array<Field, 4> &U, double eps)
{
	// compute force at U
	for (int mu = 0; mu < 4; ++mu)
		wilsonDeriv(force[mu], U, mu);

	// evolve U = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise, pRNG);
		evolve(U[mu], -eps, force[mu], std::sqrt(eps), noise, U[mu]);
	}
}

BFIntegrator::BFIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{Term(grid), Term(grid), Term(grid), Term(grid)},
      force{Field(grid), Field(grid), Field(grid), Field(grid)},
      Uprime{Field(grid), Field(grid), Field(grid), Field(grid)}, tmp{grid}
{
	pRNG.SeedFixedIntegers({seed});
}

void BFIntegrator::step(std::array<Field, 4> &U, double eps)
{
	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu], pRNG);
		wilsonDeriv(force[mu], U, mu);
		evolve(Uprime[mu], -eps, force[mu], std::sqrt(eps), noise[mu], U[mu]);
	}

	// compute force at U' and evolve U = exp(F') U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(tmp, Uprime, mu);
		force[mu] += tmp;
		evolve(U[mu], -eps * 0.5, force[mu], std::sqrt(eps), noise[mu],
		       -eps * eps * Nc / 6.0, tmp, U[mu]);
	}
}

BauerIntegrator::BauerIntegrator(Grid::GridBase *grid, int seed)
    : pRNG{grid}, noise{Term(grid), Term(grid), Term(grid), Term(grid)},
      force{grid}, Uprime{Field(grid), Field(grid), Field(grid), Field(grid)}
{
	pRNG.SeedFixedIntegers({seed});
}

void BauerIntegrator::step(std::array<Field, 4> &U, double eps)
{
	/** see https://arxiv.org/pdf/1303.3279.pdf (up to signs) */
	constexpr double k1 = 0.08578643762690485; // (2 sqrt(2) - 3) / 2;
	constexpr double k2 = 0.2928932188134524;  // (sqrt(2) - 2) / 2;
	constexpr double k5 = 0.06311327607339286; // (5 - 3 * sqrt(2)) / 12;

	// compute noise and force at U and evolve U' = exp(F) U
	for (int mu = 0; mu < 4; ++mu)
	{
		makeNoise(noise[mu], pRNG);
		wilsonDeriv(force, U, mu);
		evolve(Uprime[mu], -eps * k1, force, std::sqrt(eps) * k2, noise[mu],
		       U[mu]);
	}

	// compute force at U' and evolve U = exp(F') U
	for (int mu = 0; mu < 4; ++mu)
	{
		wilsonDeriv(force, Uprime, mu);
		evolve(U[mu], -eps, force, std::sqrt(eps), noise[mu],
		       -k5 * Nc * eps * eps, force, U[mu]);
	}
}

LandauIntegrator::LandauIntegrator(Grid::GridCartesian *grid)
    : R(grid), tmp(grid), prec(grid), fft(grid)
{

	// Idea of Fourier transform: use Laplace-operator as preconditioner.
	// Lattice laplace has known eigenvalues p^2 = 4 sum_mu sin^2(n pi / L)
	prec = 0.0;
	for (int mu = 0; mu < Nd; ++mu)
	{
		LatticeReal co(grid);
		LatticeCoordinate(co, mu);
		double L = grid->FullDimensions()[mu];
		prec += 4.0 * sin(co * (M_PI / L)) * sin(co * (M_PI / L));
	}
	RealD p2max = 4.0 * Nd; // = max(prec)

	LatticeReal def(grid);

	// set zero modes to one (yes, pretty arbitrary)
	def = 1.0;
	prec = where(prec == 0.0, def, prec);
	def = p2max;

	prec = def / prec;
}

LandauIntegrator::StepSize
LandauIntegrator::gaugecond(const std::array<Field, 4> &U) const
{
	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		if (fullAlgebra)
			tmp = logMatFast(U[mu]);
		else
			tmp = Ta(U[mu]);
		R += tmp;
		R -= Cshift(tmp, mu, -1); // NOTE: adj(A) = -A
	}

	ComplexSeries s = sum(trace(adj(R) * R)) * (1.0 / U[0]._grid->gSites());
	return toReal(s);
}

void LandauIntegrator::step(std::array<Field, 4> &U, StepSize eps)
{
	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		// turns out, both of these work fine to suppress the drift in
		// Langevin time. Though Ta() is obviously faster
		if (fullAlgebra)
			tmp = logMatFast(U[mu]);
		else
			tmp = Ta(U[mu]);
		R += tmp;
		R -= Cshift(tmp, mu, -1); // NOTE: adj(A) = -A
	}

	/** NOTE: without this projection, the non-anti-hermitian part of A will
	 * grow exponentially in Langevin time if no reunitization is done. */
	R = Ta(R);

	// Fourier acceleration
	if (fourierAccel)
	{
		fft.FFT_all_dim(tmp, R, -1); // forward Fourier (normalization 1)
		tmp *= toComplex(prec);      // why is the "toComplex" necessary here?
		fft.FFT_all_dim(R, tmp, +1); // backward Fourier (normalization 1/V)
	}
	R *= -toComplex(eps);
	R = expMatFast(R, 1.0);

	for (int mu = 0; mu < 4; ++mu)
		U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
}

LandauIntegrator::StepSize
    LandauIntegrator::stepAdaptive(std::array<Field, 4> &U, StepSize eps)
{
	std::array<Field, 4> Uprime = U;
	step(Uprime, 0.8 * eps);
	StepSize f1 = gaugecond(Uprime);

	Uprime = U;
	step(Uprime, 1.0 * eps);
	StepSize f2 = gaugecond(Uprime);

	Uprime = U;
	step(Uprime, 1.2 * eps);
	StepSize f3 = gaugecond(Uprime);

	StepSize x1 = 0.8 * eps;
	StepSize x2 = 1.0 * eps;
	StepSize x3 = 1.2 * eps;
	StepSize a =
	    x1 * x1 * (f2 - f3) + x2 * x2 * (f3 - f1) + x3 * x3 * (f1 - f2);
	StepSize b = x1 * (f2 - f3) + x2 * (f3 - f1) + x3 * (f1 - f2);
	StepSize x = 0.5 * ShiftSeries(a, -2) / ShiftSeries(b, -2);

	step(U, x);
	return x;
}
