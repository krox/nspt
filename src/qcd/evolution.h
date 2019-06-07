#ifndef QCD_EVOLUTION_H
#define QCD_EVOLUTION_H

#include "Grid/Grid.h"
#include "nspt/action.h"

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

/** NOTE: this does basic non-rescaled Langevin evolution.
 * For nice correlations, use eps=eps/beta so that the effective drift is
 * invariant of beta */
void integrateLangevin(LatticeGaugeField &U,
                       CompositeAction<LatticeGaugeField> &action,
                       GridSerialRNG &sRNG, GridParallelRNG &pRNG, double eps,
                       int sweeps, double &plaq, double &loop);

void integrateLangevinBF(LatticeGaugeField &U,
                         CompositeAction<LatticeGaugeField> &action,
                         GridSerialRNG &sRNG, GridParallelRNG &pRNG, double eps,
                         int sweeps, double &plaq, double &loop);

void integrateLangevinBauer(LatticeGaugeField &U,
                            CompositeAction<LatticeGaugeField> &action,
                            GridSerialRNG &sRNG, GridParallelRNG &pRNG,
                            double eps, int sweeps, double &plaq, double &loop);

/** Hybrid-Monte-Carlo */
void integrateHMC(LatticeGaugeField &U,
                  CompositeAction<LatticeGaugeField> &action,
                  GridSerialRNG &sRNG, GridParallelRNG &pRNG, double eps,
                  int sweeps, double &plaq, double &loop);

/** quenched SU(3) heatbath. Not actually an "integrator", but useful to have
 * the same interface */
void quenchedHeatbath(LatticeGaugeField &U,
                      CompositeAction<LatticeGaugeField> &action,
                      GridSerialRNG &sRNG, GridParallelRNG &pRNG, double,
                      int sweeps, double &plaq, double &loop);

} // namespace QCD
} // namespace Grid

#endif
