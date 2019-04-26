#ifndef NSPT_NSPT_H
#define NSPT_NSPT_H

/** utilities for NSPT. For the actual Langevin-Evolution use integrator.h  */

#include <array>

#include "Grid/Grid.h"
#include "nspt/pqcd.h"

/** one step of Landau gauge fixing (alpha=0.1 is reasonable) */
void landauStep(std::array<Grid::pQCD::LatticeColourMatrixSeries, 4> &U,
                double alpha);

/** set zero mode to zero */
void removeZero(std::array<Grid::pQCD::LatticeColourMatrixSeries, 4> &U,
                bool reunit);

#endif
