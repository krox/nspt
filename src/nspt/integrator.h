#ifndef NSPT_INTEGRATOR_H
#define NSPT_INTEGRATOR_H

#include "Grid/Grid.h"
#include "nspt/pqcd.h"
#include <array>

/**
 * Numerical integration for first order ODE/SDE. I.e. Langevin equation.
 * Not to be confused with Grid::QCD::Integrator, which is for HMC (i.e. second
 * order ODE)
 */
class Integrator
{
  public:
	// shorthands. possibly template paramters in the future
	using Field = Grid::pQCD::LatticeColourMatrixSeries;
	using Term = Grid::pQCD::LatticeColourMatrix;

	virtual ~Integrator() = default;
	virtual void step(std::array<Field, 4> &U, double eps) = 0;
};

/** first order Euler scheme */
class EulerIntegrator : public Integrator
{
	Grid::GridParallelRNG pRNG;

	// temporaries
	Term noise;
	std::array<Field, 4> force;

  public:
	EulerIntegrator(Grid::GridBase *grid, int seed);
	void step(std::array<Field, 4> &U, double eps) override;
};

/** second order "trapezoid" scheme */
class BFIntegrator : public Integrator
{
	Grid::GridParallelRNG pRNG;

	// temporaries
	std::array<Term, 4> noise;
	std::array<Field, 4> force;
	std::array<Field, 4> Uprime;
	Field tmp;

  public:
	BFIntegrator(Grid::GridBase *grid, int seed);
	void step(std::array<Field, 4> &U, double eps) override;
};

/** second order scheme from https://arxiv.org/pdf/1303.3279.pdf */
class BauerIntegrator : public Integrator
{
	Grid::GridParallelRNG pRNG;

	// temporaries
	std::array<Term, 4> noise;
	Field force;
	std::array<Field, 4> Uprime;

  public:
	BauerIntegrator(Grid::GridBase *grid, int seed);
	void step(std::array<Field, 4> &U, double eps) override;
};

class LandauIntegrator
{
	// shorthands. possibly template paramters in the future
	using Field = Grid::pQCD::LatticeColourMatrixSeries;
	using Term = Grid::pQCD::LatticeColourMatrix;
	using StepSize = Grid::pQCD::RealSeries;

	// temporaries
	mutable Field R, tmp;

	// Fourier-acceleration
	Grid::pQCD::LatticeReal prec;
	Grid::FFT fft;

  public:
	// settings
	bool fullAlgebra = true;
	bool fourierAccel = true;

	LandauIntegrator(Grid::GridCartesian *grid);
	StepSize gaugecond(const std::array<Field, 4> &U) const;
	void step(std::array<Field, 4> &U, StepSize eps);
	StepSize stepAdaptive(std::array<Field, 4> &U, StepSize eps);
};

#endif
