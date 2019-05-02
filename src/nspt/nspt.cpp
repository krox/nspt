#include "nspt/nspt.h"

using namespace Grid;
using namespace Grid::pQCD;

#include "nspt/wilson.h"

using Field = LatticeColourMatrixSeries;

void removeZero(std::array<Field, 4> &U, bool reunit)
{
	Field A{U[0]._grid};
	for (int mu = 0; mu < 4; ++mu)
	{
		A = logMatFast(U[mu]);
		ColourMatrixSeries avg = sum(A) * (1.0 / A._grid->gSites());
		A = A - avg;

		if (reunit)
			A = Ta(A);

		U[mu] = expMatFast(A, 1.0);
	}
}
