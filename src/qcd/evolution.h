#ifndef QCD_EVOLUTION_H
#define QCD_EVOLUTION_H

#include "Grid/Grid.h"
#include "nspt/action.h"

using namespace Grid;

namespace Grid {
namespace QCD {

/** compute V = exp(aX)U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            const LatticeGaugeField &U);

/** compute V = exp(aX +bY)U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            double b, const LatticeGaugeField &Y, const LatticeGaugeField &U);

/** compute V = exp(aX +bY + cZ))U. aliasing is allowed */
void evolve(LatticeGaugeField &V, double a, const LatticeGaugeField &X,
            double b, const LatticeGaugeField &Y, double c,
            const LatticeGaugeField &Z, const LatticeGaugeField &U);

/** this creates normal distribution with variance <eta^2>=2 */
void makeNoise(LatticeGaugeField &out, GridParallelRNG &pRNG);

} // namespace QCD
} // namespace Grid

class QCDIntegrator
{
  public:
	using Action = CompositeAction<QCD::LatticeGaugeField>;

	/** track basic observables during simulation */
	std::vector<double> plaq_history;
	std::vector<double> loop_history;
	int nAccept = 0;
	int nReject = 0;

	void resetStats()
	{
		plaq_history.resize(0);
		loop_history.resize(0);
		nAccept = 0;
		nReject = 0;
	}

	double acceptance() const { return double(nAccept) / (nAccept + nReject); }

	virtual ~QCDIntegrator() = default;
	virtual void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	                 GridParallelRNG &pRNG, int sweeps) = 0;

  protected:
	// to be used after every sweep
	void trackObservables(QCD::LatticeGaugeField &U)
	{
		plaq_history.push_back(QCD::ColourWilsonLoops::avgPlaquette(U));
		loop_history.push_back(
		    real(QCD::ColourWilsonLoops::avgPolyakovLoop(U)));
	}
};

/** NOTE: this does basic non-rescaled Langevin evolution.
 * For nice correlations, use eps=eps/beta so that the effective drift is
 * invariant of beta */
class LangevinEuler : public QCDIntegrator
{
  public:
	Action &action;
	double delta;

	LangevinEuler(Action &action, double eps)
	    : action(action), delta(eps / action.beta)
	{}

	void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	         GridParallelRNG &pRNG, int sweeps) override;
};

class LangevinBF : public QCDIntegrator
{
  public:
	Action &action;
	double delta;

	LangevinBF(Action &action, double eps)
	    : action(action), delta(eps / action.beta)
	{}

	void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	         GridParallelRNG &pRNG, int sweeps) override;
};

class LangevinBauer : public QCDIntegrator
{
  public:
	Action &action;
	double delta;

	LangevinBauer(Action &action, double eps)
	    : action(action), delta(eps / action.beta)
	{}
	void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	         GridParallelRNG &pRNG, int sweeps) override;
};

/** Hybrid-Monte-Carlo */
class HMC : public QCDIntegrator
{
  public:
	Action &action;
	double delta;
	bool metropolis;

	HMC(Action &action, double eps, bool metropolis)
	    : action(action), delta(eps / std::sqrt(action.beta)),
	      metropolis(metropolis)
	{}

	void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	         GridParallelRNG &pRNG, int sweeps) override;
};

/** quenched SU(3) heatbath. Not actually an "integrator", but useful to have
 * the same interface */
class Heatbath : public QCDIntegrator
{
  public:
	double beta;

	Heatbath(Action &action) : beta(action.beta)
	{
		assert(action.gauge_action == "wilson");
		assert(action.fermion_action == "");
	}

	void run(QCD::LatticeGaugeField &U, GridSerialRNG &sRNG,
	         GridParallelRNG &pRNG, int sweeps) override;
};

std::unique_ptr<QCDIntegrator> makeQCDIntegrator(QCDIntegrator::Action &action,
                                                 const json &j);

#endif
