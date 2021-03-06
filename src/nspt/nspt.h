#ifndef NSPT_NSPT_H
#define NSPT_NSPT_H

/** utilities for NSPT. For the actual Langevin-Evolution use integrator.h  */

#include <array>

#include "Grid/Grid.h"
#include "nspt/pqcd.h"

/** set zero mode to zero */
void removeZero(std::array<Grid::pQCD::LatticeColourMatrixSeries, 4> &U,
                bool reunit);

#endif
