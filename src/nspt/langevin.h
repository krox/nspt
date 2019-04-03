#ifndef NSPT_LANGEVIN_H
#define NSPT_LANGEVIN_H

#include "Grid/Grid.h"

#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"

/** Langevin evolution for (quenched) SU(3) lattice theory */
class Langevin
{
  public:
	Grid::GridCartesian *grid;
	Grid::GridParallelRNG pRNG;
	Grid::GridSerialRNG sRNG;

	bool flipNoise = false;

	using Field = Grid::pQCD::LatticeColourMatrix;

	// gauge config
	std::array<Field, 4> U;

	// temporary
	std::array<Field, 4> Uprime;

	// Set config to Unit
	Langevin(std::vector<int> latt, int seed);

	void makeNoise(Field &out, double eps);

	/** one step of (gluonic) Langevin evolution */
	void evolveStep(double eps, double beta); // first-order Euler scheme
	void evolveStepImproved(double eps,
	                        double beta); // second-order "BF" scheme
	void evolveStepBauer(double eps,
	                     double beta); // second-order "Bauer" scheme
};

#endif
