#ifndef NSPT_LATTICE_IO_H
#define NSPT_LATTICE_IO_H

#include "Grid/Grid.h"
#include "nspt/grid_utils.h"
#include "nspt/pqcd.h"

using namespace Grid;

void writeConfig(const std::string &filename, QCD::LatticeGaugeField &U);

std::vector<int> getGridFromFile(const std::string &filename);
void readConfig(const std::string &filename, QCD::LatticeGaugeField &U);

std::vector<int> getGridFromFileOpenQCD(const std::string &filename);
void readConfigOpenQCD(const std::string &filename, QCD::LatticeGaugeField &U);

#endif
