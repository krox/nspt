#ifndef NSPT_NSPT_H
#define NSPT_NSPT_H

#include "Grid/Grid.h"

#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"

/** Perturbative Langevin evolution for (quenched) SU(3) lattice theory */
class LangevinPert
{
  public:
	Grid::GridCartesian *grid;
	Grid::GridParallelRNG pRNG;
	Grid::GridSerialRNG sRNG;

	bool flipNoise = false;

	// Fields are expanded in beta^-0.5
	using Field = Grid::pQCD::LatticeColourMatrixSeries;
	using FieldTerm = Grid::pQCD::LatticeColourMatrix;

	// gauge config
	std::array<Field, 4> U;

	// temporary
	std::array<Field, 4> Uprime;

	// Set config to Unit
	LangevinPert(std::vector<int> latt, int seed);

	void makeNoise(FieldTerm &out, double eps);

	/** one step of (gluonic) Langevin evolution */
	void evolveStep(double eps);         // first-order Euler scheme
	void evolveStepImproved(double eps); // second-order "BF" scheme
	void evolveStepBauer(double eps);    // second-order "Bauer" scheme

	/** one step of Landau gauge fixing */
	void landauStep(double alpha);

	/** compute the algebra elements A=log(U) */
	std::array<Field, 4> algebra();

	/** set zero mode to zero */
	void zmreg(bool reunit);
};

#endif
