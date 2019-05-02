#include "nspt/algebra_observables.h"

using namespace Grid;
using namespace pQCD;

void AlgebraObservables::measure(const std::array<Field, 4> &A)
{
	// lattice volume
	double V = A[0]._grid->gSites();

	// trace/herm/norm of field
	{
		RealSeries tmp1 = 0.0;
		RealSeries tmp2 = 0.0;
		RealSeries tmp3 = 0.0;

		for (int mu = 0; mu < 4; ++mu)
		{
			tmp1 += (1.0 / V) * norm2_series(trace(A[mu]));
			tmp2 += (1.0 / V) * norm2_series(Field(A[mu] + adj(A[mu])));
			tmp3 += norm2_series(A[mu]) * (1.0 / V);
		}

		traceA.push_back(asSpan(tmp1));
		hermA.push_back(asSpan(tmp2));
		normA.push_back(asSpan(tmp3));
	}

	// norm of average field (i.e. zero mode)
	avgAx.push_back(asSpan((1.0 / V) * norm2_series(sum(A[0]))));
	avgAy.push_back(asSpan((1.0 / V) * norm2_series(sum(A[1]))));
	avgAz.push_back(asSpan((1.0 / V) * norm2_series(sum(A[2]))));
	avgAt.push_back(asSpan((1.0 / V) * norm2_series(sum(A[3]))));

	// gauge condition
	LatticeColourMatrixSeries w = A[0] - Cshift(A[0], 0, -1);
	w += A[1] - Cshift(A[1], 1, -1);
	w += A[2] - Cshift(A[2], 2, -1);
	w += A[3] - Cshift(A[3], 3, -1);
	gaugeCond.push_back(asSpan((1.0 / V) * toReal(sum(trace(adj(w) * w)))));
}
