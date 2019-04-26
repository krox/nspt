#include "nspt/nspt.h"

using namespace Grid;
using namespace Grid::pQCD;

#include "nspt/wilson.h"

using Field = LatticeColourMatrixSeries;

void landauStep(std::array<Field, 4> &U, double alpha)
{
	LatticeColourMatrixSeries R(U[0]._grid);

	R = 0.0;
	for (int mu = 0; mu < 4; ++mu)
	{
		// turns out, both of these work fine to suppress the drift in
		// Langevin time. Though Ta() is obviously faster
		Field Amu = logMatFast(U[mu]);
		// Field Amu = Ta(U[mu]);

		R += Amu;
		R -= Cshift(Amu, mu, -1); // NOTE: adj(A) = -A
	}

	/** NOTE: without this projection, the non-anti-hermitian part of A will
	 * grow exponentially in time if no reunitization is done. */
	R = Ta(R);

	R = expMatFast(R, -alpha);

	for (int mu = 0; mu < 4; ++mu)
		U[mu] = R * U[mu] * Cshift(adj(R), mu, 1);
}

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
