
#ifndef ESTIMATEPORTFOLIO_H
#define ESTIMATEPORTFOLIO_H
#include <map>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/normal_dist.hpp>


/* \file EstimatePortfolio.h
 *       Taken from 'Monte Carlo for high-dimensional degenerated Semi Linear and Full Non Linear PDEs' by X Warin
 *          Semi analytical solution given by
 *         \f[
 *            v(t,x,y^1,.., y^d)=-e^{-\eta x} \E[\prod_{i=1}^d \exp\left(-\frac{1}{2}\int_t^T \frac{(\mu^i)^2}{\tilde Y^i_s}ds \right) ]
 *         \f]

 * \author Xavier Warin
 */


namespace libflow
{

/// \brief Semi analytical solution
///  \param  p_muS     trend of the asset
///  \param  p_meanv   mean reverting of CIR for vol
///  \param  p_revertv assymptotic  value of the vol
///  \param  p_sigv    volalitlity of volatility
///  \param  p_sigs    volalitlity of the asset
///  \param  p_T       maturity
///  \param  p_eta     risk aversion
///  \param  p_y0      initial value of the volatility
///  \param  p_x0      initial value of the wealth
///  \param  p_nbSim   number of Monte Carlo simulations
///  \param  p_nbStep  number of time step for Euler scheme
///  \return optimized value
///  Milstein Implicit Scheme (Kahl Jackel)  for positivity of CIR
template< int NDIM>
double estimatePortfolio(double p_muS, double p_meanv, double p_revertv, double    p_sigv, double p_sigS,
                         double p_T, double p_eta, double p_y0, double p_x0,  int p_nbSim, int p_nbStep)
{
    assert(pow(p_sigv, 2.) < 4 * p_meanv * p_revertv);
    boost::mpi::communicator world;
    trng::yarn2 gen;
    gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);
    int nbSimLoc = p_nbSim / world.size();
    p_nbSim = nbSimLoc * world.size();

    double step = p_T / p_nbStep ;
    double sqrp = sqrt(step);
    double ymed = 0. ;
    for (int is = 0 ; is < nbSimLoc; ++is)
    {
        Eigen::Array<double, NDIM, 1> y = Eigen::Array<double, NDIM, 1>::Constant(p_y0) ;
        Eigen::Array<double, NDIM, 1>  yint = Eigen::Array<double, NDIM, 1>::Zero() ; //integrale
        for (int i = 0 ; i < p_nbStep; ++i)
        {
            for (int id = 0; id  < NDIM; ++id)
            {
                double xrand = normal(gen);
                // evolution de y
                y(id) = (y(id) + p_meanv * p_revertv * step +
                         p_sigv * sqrt(y(id)) * xrand * sqrp + 1. / 2. * (0.5 * p_sigv * p_sigv) * step * (xrand * xrand - 1.)) / (1 + p_revertv * step);
                yint(id) += 0.5 * step * p_muS * p_muS / (p_sigS * p_sigS * y(id));
                assert(y(id) > 0.);
            }
        }
        double sol = -exp(- p_eta * p_x0) * exp(-yint.sum());
        ymed += sol;
    }
    double yRet = 0.;
    boost::mpi::all_reduce(world, ymed, yRet, std::plus<double>());
    yRet /= p_nbSim;
    return yRet;
}

}
#endif
