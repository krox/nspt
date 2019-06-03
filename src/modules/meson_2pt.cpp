#include "modules/meson_2pt.h"

#include "nspt/grid_utils.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include <Grid/Grid.h>

using namespace Grid;
using namespace util;

using GaugeField = QCD::LatticeLorentzColourMatrix;
using GaugeMat = QCD::LatticeColourMatrix;
using FermionField = QCD::LatticeSpinColourVector;
using PropagatorField = QCD::LatticeSpinColourMatrix;
using LatticeComplex = QCD::LatticeComplex;
using QCD::Gamma;

static void makePhase(LatticeComplex &phase, const std::vector<int> &mom,
                      const std::vector<int> &origin = {0, 0, 0, 0})
{
	auto grid = phase._grid;
	assert(mom.size() == grid->FullDimensions().size());
	phase = 0.0;
	LatticeComplex coor(grid);
	for (int mu = 0; mu < (int)mom.size(); ++mu)
	{
		LatticeCoordinate(coor, mu);
		double c = M_PI * 2.0 / grid->FullDimensions()[mu] * mom[mu];
		phase += c * (coor - (double)origin[mu]);
	}
	phase = exp(phase * Complex(0.0, 1.0));
}

static std::string trim(const std::string &s)
{
	auto r = s;
	while (r.size() && r.back() == ' ')
		r = r.substr(0, r.size() - 1);
	return r;
}

void MMeson2pt::run(Environment &env)
{
	// solver parameters (maybe put them in json?)
	double solver_tol = 1.0e-8;
	int solver_max_iter = 5000;
	int solver_restart = 25;

	// gauge field
	GaugeField &U = env.store.get<GaugeField>(params.field);
	GridCartesian *grid = env.getGrid(U._grid->FullDimensions());
	GridRedBlackCartesian *gridRB = env.getGridRB(U._grid->FullDimensions());

	// fermion solver
	std::unique_ptr<QCD::WilsonCloverFermion<QCD::WilsonImplR>> fermOperator;
	std::unique_ptr<OperatorFunction<FermionField>> fermSolver;

	if (params.fermion_action == "wilson_clover_nf2")
	{
		// NOTE: bare mass is typically negative
		double mass = 0.5 * (1.0 / params.kappa_light - 8.0);
		double csw = params.csw;
		fmt::print("mass = {}, kappa = {}, csw = {}\n", mass,
		           params.kappa_light, csw);

		fermOperator = std::make_unique<QCD::WilsonCloverFermionR>(
		    U, *grid, *gridRB, mass, csw, csw);

		fermSolver = std::make_unique<GeneralisedMinimalResidual<FermionField>>(
		    solver_tol, solver_max_iter, solver_restart, true);
	}
	else
		throw std::runtime_error("unknown fermion action");

	// build the propagator
	PropagatorField prop(grid);
	FermionField source(grid);
	FermionField psi(grid);
	source = 0.0;
	for (int alpha = 0; alpha < 4; ++alpha)
		for (int a = 0; a < 3; ++a)
		{
			// fmt::print("inverting {} {}\n", alpha, a);
			// build source
			QCD::SpinColourVector local = zero;
			local()(alpha)(a) = 1.0;
			pokeSite(local, source, params.source);

			// invert (NOTE: this solves M^-1, even though it says "MdagM")
			MdagMLinearOperator<QCD::WilsonFermion<QCD::WilsonImplR>,
			                    FermionField>
			    op(*fermOperator);
			(*fermSolver)(op, source, psi);

			// inject into propagator
			parallel_for(int ss = 0; ss < grid->oSites(); ss++)
			{
				for (int beta = 0; beta < 4; ++beta)
					for (int b = 0; b < 3; ++b)
						prop._odata[ss]()(beta, alpha)(b, a) =
						    psi._odata[ss]()(beta)(b);
			}
		}

	// backward propagator by gamma5 hermiticity
	PropagatorField S = Gamma(Gamma::Algebra::Gamma5) * adj(prop) *
	                    Gamma(Gamma::Algebra::Gamma5);

	DataFile file;

	if (primaryTask() && params.filename != "")
		file = DataFile::create(params.filename);

	for (auto &gi : Gamma::gall)
	{
		if (primaryTask())
			fmt::print("Contracting {}\n", Gamma::name[gi.g]);

		// contract with gamma matrix (gi=Gamma5 == Pion)
		LatticeComplex c2pt = trace(Gamma(gi) * prop * Gamma(gi) * S);

		for (auto &mom : params.sink_mom_list)
		{
			// phase factor for momentum projection
			assert(mom.size() == 3);
			std::vector<int> mom4 = mom; // minus sign???
			mom4.push_back(0);
			LatticeComplex phase(grid);
			makePhase(phase, mom4, params.source);

			std::vector<QCD::TComplexD> buf(grid->FullDimensions()[3]);
			sliceSum(LatticeComplex(phase * c2pt), buf, 3);

			// write out result per timeslice, shifted to source=0
			if (primaryTask() && params.filename != "")
			{
				std::vector<double> c2pt_real;
				std::vector<double> c2pt_imag;
				for (int i = 0; i < (int)buf.size(); ++i)
				{
					int t = (i + params.source[3]) % grid->FullDimensions()[3];
					c2pt_real.push_back(real(TensorRemove(buf[t])));
					c2pt_imag.push_back(imag(TensorRemove(buf[t])));
				}

				std::string chan =
				    fmt::format("{}_{}{}{}", trim(Gamma::name[gi.g]), mom[0],
				                mom[1], mom[2]);
				file.createData(chan + "_real", c2pt_real);
				file.createData(chan + "_imag", c2pt_real);

				if (chan == "Gamma5_000")
				{
					for (int i = 1; i < grid->FullDimensions()[3] - 1; ++i)
					{
						// symmetrized effective mass
						double m =
						    acosh(0.5 * (c2pt_real[i - 1] + c2pt_real[i + 1]) /
						          c2pt_real[i]);
						fmt::print("m_pi,eff({}) = {} ( Lmpi = {} )\n", i, m,
						           m * grid->FullDimensions()[0]);
					}
				}
			}
		}
	}
}
