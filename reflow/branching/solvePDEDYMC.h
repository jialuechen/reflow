
#ifndef SOLVEPDEDYMC_H
#define SOLVEPDEDYMC_H
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <trng/normal_dist.hpp>

/** \file SolvePDEDYMC.h
 *  \brief Permits to solve High dimensional PDE using methodology
 *         'Nesting Monte Carlo for high-dimensional Non Linear PDEs' by X Warin
 *         Equation
 *         \f[
 *             (-\partial_t u-{\cal L} u)(t,x)   = f(t,x,u(t,x), Du(t,x)),
 *         \f]
 *         with
 *         \f[
 *             u_T =g
 *         \f]
 *         and
 *         \f[
 *           {\cal L} u(t,x) := \mu Du(t,x) +  \frac{1}{2} \sigma \sigma^{\top} \!:\! D^2 u(t,x)
 *         \f]
 *         Special case where the non linearity satisfies \f$ f(t,x, u) \f$
 *         Three versions
 *             - Euler scheme
 *             - exact simulation of the PDE simulated with :
 *              \f[
 *                   X_{t+dt} =  A(t,dt)  X_t + B(t,dt) + C(t,dt) g
 *               \f]
 *               where $g$ is a unit centered gaussian vector.
 *             - Constant  coefficient for SDE
 *              \f[
 *                   X_{t+dt} =   X_t + \mu  + \sigma g
 *               \f]
 *             .
 *   \author Xavier Warin
 */

namespace reflow
{
/// \brief PDE resolution for  one step and one simulation with an Euler scheme
///        Suppose that the drift and diffusion are given by functions
/// \param p_initVal           initial values for all points
/// \param p_mu                drift vector function
/// \param p_sigma             diffusion matrix function
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               TRNG random number generator
/// \param p_g                 Terminal function
/// \param p_stepEuler         Euler step
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N, int P, class SwitchDistrib, class TRNGGenerator>
class SolveOneStepDyEuler
{
public :

    inline std::pair< Eigen::Array<double, P, 1>, Eigen::Array< double, N, P>  >   operator()(const Eigen::Matrix<double, N, P>   &p_initVal,
            const std::function< Eigen::Matrix<double, N, 1> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_mu,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_sigma,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const double &p_stepEuler,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        // one step for position
        double dt = std::max(std::min(p_lawSwitch(p_gen), p_T - p_timeInit), 1e-20);
        // number Euler steps
        int nbEulerSteps = std::max(static_cast<int>(dt / p_stepEuler), 1);
        double dtLoc = dt / nbEulerSteps;
        double sqrtdtLoc = sqrt(dtLoc);

        // store SDE solution
        Eigen::Matrix<double, N, 2 * P>  xProc;
        xProc.leftCols(P) = p_initVal;
        xProc.rightCols(P) = p_initVal; // antithetic

        // keep first gauss
        Eigen::Matrix< double, N, 1> gaussKeep;
        for (int id = 0; id < N; ++id)
            gaussKeep(id) = p_normal(p_gen);

        // sigma first step
        Eigen::Array< Eigen::Matrix<double, N, N>, P, 1> sigmaInv ;

        for (int i = 0; i < P ; ++i)
        {
            Eigen::Matrix<double, N, 1> mu = p_mu(p_timeInit, p_initVal.col(i));
            Eigen::Matrix<double, N, N> sigma = p_sigma(p_timeInit, p_initVal.col(i));
            xProc.col(i) +=  mu * dtLoc + sigma * sqrtdtLoc * gaussKeep;
            xProc.col(i + P) +=  mu * dtLoc - sigma * sqrtdtLoc * gaussKeep;
            sigmaInv(i) = sigma.inverse();
        }

        // nest on dates
        double tSuiv = p_timeInit;
        for (int j = 1; j < nbEulerSteps; ++j)
        {
            tSuiv += dtLoc;
            Eigen::Matrix< double, N, 1> gauss;
            for (int id = 0; id < N; ++id)
                gauss(id) = p_normal(p_gen);
            for (int i = 0; i < P; ++i)
            {
                xProc.col(i) +=  p_mu(tSuiv, xProc.col(i)) * dtLoc + p_sigma(tSuiv, xProc.col(i)) * sqrtdtLoc * gauss;
                xProc.col(i + P) +=  p_mu(tSuiv, xProc.col(i + P)) * dtLoc + p_sigma(tSuiv, xProc.col(i + P)) * sqrtdtLoc * gauss;
            }
        }

        // for return
        Eigen::Array< double, P, 1> ret ;
        Eigen::Array< double, N, P> retGrad;

        // new date
        double newDate = p_timeInit + dt;

        // test
        if ((newDate >=  p_T - 1e-5) || (p_depth == p_nbSim.size()))
        {
            double unSurCDF = 1. / (1. - p_lawSwitch.cdf(dt));
            for (int i = 0; i < P ; ++i)
            {
                double finalG = p_g(xProc.col(i));
                double finalGAnt = p_g(xProc.col(i + P));
                ret(i) = 0.5 * (finalG + finalGAnt) * unSurCDF;
                Eigen::Matrix<double, N, 1> sigInvGauss = sigmaInv(i).transpose() * gaussKeep;
                retGrad.col(i) = 0.5 * (finalG - finalGAnt) * unSurCDF / sqrtdtLoc * sigInvGauss.array();
            }
        }
        else
        {
            int nbsimul = p_nbSim[p_depth];
            double lawSwitchPdf = p_lawSwitch.pdf(dt) ;
            Eigen::Array<double, 2 * P, 1 > yVal = Eigen::Array<double, 2 * P, 1 >::Zero();
            Eigen::Array<double, N, 2 * P>  zVal =  Eigen::Array<double, N, 2 * P >::Zero();
            // nesting recursion
            for (int is = 0 ; is < nbsimul; ++is)
            {
                std::pair< Eigen::Array<double, 2 * P, 1 >, Eigen::Array<double, N, 2 * P > >  solAndGrad =
                    SolveOneStepDyEuler<N, 2 * P, SwitchDistrib, TRNGGenerator>()(xProc, p_mu, p_sigma,  p_f, newDate, p_T, p_lawSwitch, p_normal,
                            p_gen, p_g,  p_stepEuler, p_nbSim,
                            p_depth + 1);
                yVal += solAndGrad.first;
                zVal += solAndGrad.second;
            }
            yVal /= nbsimul;
            zVal /= nbsimul;
            for (int i = 0; i < P; ++i)
            {
                double obstacle = p_f(newDate, xProc.col(i), yVal(i), zVal.col(i)) ;
                double obstacleAnt =  p_f(newDate, xProc.col(i + P), yVal(i + P), zVal.col(i + P));
                ret(i) = 0.5 * (obstacle + obstacleAnt) / lawSwitchPdf;
                Eigen::Matrix<double, N, 1> sigInvGauss = sigmaInv(i).transpose() * gaussKeep;
                retGrad.col(i) = 0.5 * (obstacle - obstacleAnt) / sqrtdtLoc * sigInvGauss / lawSwitchPdf;
            }
        }
        return std::make_pair(ret, retGrad);
    }
};

/// \brief PDE resolution for  one step and one simulation with an Euler scheme
///        Suppose that the drift and diffusion are given by functions
//         Specialized template : permits to stop the template recursion
/// \param p_initVal           initial values for all points
/// \param p_mu                drift vector function
/// \param p_sigma             diffusion matrix function
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_stepEuler         Euler step
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N,   class SwitchDistrib, class TRNGGenerator >
class SolveOneStepDyEuler<N, 64, SwitchDistrib, TRNGGenerator >
{
public :

    inline std::pair< Eigen::Array<double, 64, 1>, Eigen::Array< double, N, 64>  >   operator()(const Eigen::Matrix<double, N, 64>   &p_initVal,
            const std::function< Eigen::Matrix<double, N, 1> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_mu,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_sigma,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const double &p_stepEuler,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        Eigen::Array<double, 64, 1> ret ;
        Eigen::Array< double, N, 64> retGrad;
        return std::make_pair(ret, retGrad);
    }
};


/// \brief PDE resolution for Euler Scheme
/// \param p_mu                drift vector
/// \param p_sigma             volatility matrix
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_point             initial point  where to calculate the solution
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_stepEuler         step Euler
/// \param p_gen               TRNG random number generator
/// \return values std /sqrt(n) for val
template< int N, class SwitchDistrib, class TRNGGenerator  >
std::tuple< double, Eigen::Array<double, N, 1>, double,  Eigen::Array<double, N, 1> >   solvePDEDYMCEuler(const std::function< Eigen::Matrix<double, N, 1> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_mu,
        const std::function< Eigen::Matrix<double, N, N> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_sigma,
        const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
        const Eigen::Array<double, N, 1>   &p_point,
        const double   &p_timeInit,
        const double   &p_T,
        SwitchDistrib &p_lawSwitch,
        std::function< double (const Eigen::Matrix<double, N, 1>&)>   &p_g,
        const std::vector<int> &p_nbSim,
        const double &p_stepEuler,
        TRNGGenerator &p_gen)
{

    // mpi
    boost::mpi::communicator world;
    //  random generator
    p_gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);

    // global simulation number
    int nbSimul = p_nbSim[0];
    int nbSimulProc = nbSimul / world.size();

    // store PDE value and std associated for first level
    double  val = 0.;
    double valSq =  0.;
    Eigen::Array<double, N, 1>  valGrad = Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>  valGradSq = Eigen::Array<double, N, 1>::Zero();
    int depth = 1;

    //#pragma omp parallel for schedule(dynamic,10)
    for (int is = 0; is <   nbSimulProc ; ++is)
    {
        if (world.rank() == 0)
            if (is % 1000 == 0)
                std::cout << " is " << is <<   std::endl ;
        std::pair<Eigen::Array<double, 1, 1>, Eigen::Array<double, N, 1> >  solLocAndGrad =  SolveOneStepDyEuler<N, 1, SwitchDistrib, TRNGGenerator>()(p_point,  p_mu, p_sigma, p_f, p_timeInit, p_T, p_lawSwitch, normal, p_gen, p_g, p_stepEuler, p_nbSim, depth);
        val += solLocAndGrad.first(0);
        valSq += solLocAndGrad.first(0) * solLocAndGrad.first(0);
        valGrad += solLocAndGrad.second;
        valGradSq += solLocAndGrad.second * solLocAndGrad.second;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, val, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, valSq, valSqRet, std::plus<double>());
    Eigen::Array<double, N, 1>  valGradRet =  Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>   valGradSqRet = Eigen::Array<double, N, 1>::Zero();
    boost::mpi::all_reduce(world, valGrad.data(), N, valGradRet.data(), std::plus<double>());
    boost::mpi::all_reduce(world, valGradSq.data(), N, valGradSqRet.data(), std::plus<double>());
    int nbSim = nbSimulProc * world.size();
    valRet /= nbSim;
    valSqRet /= nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / nbSim);
    valGradRet /= nbSim;
    valGradSqRet /= nbSim   ;
    valGradSqRet = ((valGradSqRet - valGradRet * valGradRet) / nbSim).sqrt();
    return std::make_tuple(valRet, valGradRet, valSqRet, valGradSqRet);
}



/// \brief PDE resolution for  one step and one simulation with an exact scheme
/// \param p_initVal           initial values for all points
/// \param p_A                 multiplicative factor for \f$ X_{t} \f$
/// \param P_B                 constant term in SDE
/// \param P_C                 volatility
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N, int P, class SwitchDistrib, class TRNGGenerator >
class SolveOneStepDyExact
{
public :
    inline std::pair< Eigen::Array<double, P, 1>, Eigen::Array< double, N, P>  >   operator()(const Eigen::Matrix<double, N, P>   &p_initVal,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const  double &) >   &p_A,
            const std::function< Eigen::Matrix<double, N, 1> (const double &, const  double &) >   &p_B,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const   double &) >     &p_C,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        // one step for position
        double dt = std::max(std::min(p_lawSwitch(p_gen), p_T - p_timeInit), 1e-20);
        double sqrtdt = sqrt(dt);

        // store SDE solution
        Eigen::Matrix<double, N, 2 * P>  xProc;

        // keep first gauss
        Eigen::Matrix< double, N, 1> gaussKeep;
        for (int id = 0; id < N; ++id)
            gaussKeep(id) = p_normal(p_gen);

        // sigma first step
        Eigen::Matrix<double, N, N> A = p_A(p_timeInit, dt);
        Eigen::Matrix<double, N, 1> B = p_B(p_timeInit, dt);
        Eigen::Matrix<double, N, N> C = p_C(p_timeInit, dt);
        // For Malliavin weights
        Eigen::Matrix<double, N, N>  sigmaInv = (A.inverse() * C).inverse();
        Eigen::Matrix<double, N, N>  sigmaInvTrans = (A.inverse() * C).inverse();
        Eigen::Matrix<double, N, 1> sigInvGauss = sigmaInvTrans * gaussKeep;

        for (int i = 0; i < P ; ++i)
        {
            xProc.col(i) =  A * p_initVal.col(i) + B + C * gaussKeep;
            xProc.col(i + P) = A * p_initVal.col(i) + B  - C * gaussKeep;
        }

        // for return
        Eigen::Array< double, P, 1> ret ;
        Eigen::Array< double, N, P> retGrad;

        // new date
        double newDate = p_timeInit + dt;

        // test
        if ((newDate >=  p_T - 1e-5) || (p_depth == p_nbSim.size()))
        {
            double unSurCDF = 1. / (1. - p_lawSwitch.cdf(dt));
            for (int i = 0; i < P ; ++i)
            {
                double finalG = p_g(xProc.col(i));
                double finalGAnt = p_g(xProc.col(i + P));
                ret(i) = 0.5 * (finalG + finalGAnt) * unSurCDF;
                retGrad.col(i) = 0.5 * (finalG - finalGAnt) * unSurCDF * sigInvGauss.array();
            }
        }
        else
        {
            int nbsimul = p_nbSim[p_depth];
            double lawSwitchPdf = p_lawSwitch.pdf(dt) ;
            Eigen::Array<double, 2 * P, 1 > yVal = Eigen::Array<double, 2 * P, 1 >::Zero();
            Eigen::Array<double, N, 2 * P>  zVal =  Eigen::Array<double, N, 2 * P >::Zero();
            // nesting recursion
            for (int is = 0 ; is < nbsimul; ++is)
            {
                std::pair< Eigen::Array<double, 2 * P, 1 >, Eigen::Array<double, N, 2 * P > >  solAndGrad =
                    SolveOneStepDyExact<N, 2 * P, SwitchDistrib, TRNGGenerator>()(xProc, p_A, p_B, p_C,  p_f, newDate, p_T, p_lawSwitch, p_normal,
                            p_gen, p_g,  p_nbSim,
                            p_depth + 1);
                yVal += solAndGrad.first;
                zVal += solAndGrad.second;
            }
            yVal /= nbsimul;
            zVal /= nbsimul;
            for (int i = 0; i < P; ++i)
            {
                double obstacle = p_f(newDate, xProc.col(i), yVal(i), zVal.col(i)) ;
                double obstacleAnt =  p_f(newDate, xProc.col(i + P), yVal(i + P), zVal.col(i + P));
                ret(i) = 0.5 * (obstacle + obstacleAnt) / lawSwitchPdf;
                retGrad.col(i) = 0.5 * (obstacle - obstacleAnt) * sigInvGauss / lawSwitchPdf;
            }
        }
        return std::make_pair(ret, retGrad);
    }
};

/// \brief PDE resolution for  one step and one simulation with an exact scheme
///        Specialized template : permits to stop the template recursion
/// \param p_initVal           initial values for all points
/// \param p_A                 multiplicative factor for \f$ X_{t} \f$
/// \param P_B                 constant term in SDE
/// \param P_C                 volatility
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N,  class SwitchDistrib, class TRNGGenerator>
class SolveOneStepDyExact<N, 64, SwitchDistrib, TRNGGenerator>
{
public :
    inline std::pair< Eigen::Array<double, 64, 1>, Eigen::Array< double, N, 64>  >   operator()(const Eigen::Matrix<double, N, 64>   &p_initVal,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const  double &) >   &p_A,
            const std::function< Eigen::Matrix<double, N, 1> (const double &, const  double &) >   &p_B,
            const std::function< Eigen::Matrix<double, N, N> (const double &, const   double &) >     &p_C,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        Eigen::Array<double, 64, 1> ret ;
        Eigen::Array< double, N, 64> retGrad;
        return std::make_pair(ret, retGrad);
    }
};


/// \brief PDE resolution for  exact simulation
/// \param p_A                 multiplicative factor for \f$ X_{t} \f$
/// \param P_B                 constant term in SDE
/// \param P_C                 volatility
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_point             initial point  where to calculate the solution
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return values std /sqrt(n) for val
template< int N, class SwitchDistrib, class TRNGGenerator >
std::tuple< double, Eigen::Array<double, N, 1>, double,  Eigen::Array<double, N, 1> >   solvePDEDYMCExact(
    const std::function< Eigen::Matrix<double, N, N> (const double &, const  double &) >   &p_A,
    const std::function< Eigen::Matrix<double, N, 1> (const double &, const  double &) >   &p_B,
    const std::function< Eigen::Matrix<double, N, N> (const double &, const   double &) >     &p_C,
    const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
    const Eigen::Array<double, N, 1>   &p_point,
    const double   &p_timeInit,
    const double   &p_T,
    SwitchDistrib &p_lawSwitch,
    std::function< double (const Eigen::Matrix<double, N, 1>&)>   &p_g,
    const std::vector<int> &p_nbSim,
    TRNGGenerator &p_gen)
{

    // mpi
    boost::mpi::communicator world;
    //  random generator
    p_gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);

    // global simulation number
    int nbSimul = p_nbSim[0];
    int nbSimulProc = nbSimul / world.size();

    // store PDE value and std associated for first level
    double  val = 0.;
    double valSq =  0.;
    Eigen::Array<double, N, 1>  valGrad = Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>  valGradSq = Eigen::Array<double, N, 1>::Zero();
    int depth = 1;

    //#pragma omp parallel for schedule(dynamic,10)
    for (int is = 0; is <   nbSimulProc ; ++is)
    {
        if (world.rank() == 0)
            if (is % 1000 == 0)
                std::cout << " is " << is <<   std::endl ;
        std::pair<Eigen::Array<double, 1, 1>, Eigen::Array<double, N, 1> >  solLocAndGrad =  SolveOneStepDyExact<N, 1, SwitchDistrib, TRNGGenerator>()(p_point,  p_A, p_B, p_C, p_f, p_timeInit, p_T, p_lawSwitch, normal, p_gen, p_g, p_nbSim, depth);
        val += solLocAndGrad.first(0);
        valSq += solLocAndGrad.first(0) * solLocAndGrad.first(0);
        valGrad += solLocAndGrad.second;
        valGradSq += solLocAndGrad.second * solLocAndGrad.second;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, val, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, valSq, valSqRet, std::plus<double>());
    Eigen::Array<double, N, 1>  valGradRet =  Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>   valGradSqRet = Eigen::Array<double, N, 1>::Zero();
    boost::mpi::all_reduce(world, valGrad.data(), N, valGradRet.data(), std::plus<double>());
    boost::mpi::all_reduce(world, valGradSq.data(), N, valGradSqRet.data(), std::plus<double>());
    int nbSim = nbSimulProc * world.size();
    valRet /= nbSim;
    valSqRet /= nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / nbSim);
    valGradRet /= nbSim;
    valGradSqRet /= nbSim   ;
    valGradSqRet = ((valGradSqRet - valGradRet * valGradRet) / nbSim).sqrt();
    return std::make_tuple(valRet, valGradRet, valSqRet, valGradSqRet);
}




/// \brief PDE resolution for  one step and one simulation with for an SDE with constant coefficients
/// \param p_initVal           initial values for all points
/// \param p_mu                drift vector
/// \param p_sigma             diffusion matrix
/// \param p_sigmaInv          inverse of diffusion matrix
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N, int P, class SwitchDistrib, class TRNGGenerator>
class SolveOneStepDyConst
{
public :

    inline std::pair< Eigen::Array<double, P, 1>, Eigen::Array< double, N, P>  >   operator()(const Eigen::Matrix<double, N, P>   &p_initVal,
            const Eigen::Matrix<double, N, 1>    &p_mu,
            const Eigen::Matrix<double, N, N>    &p_sigma,
            const Eigen::Matrix<double, N, N>    &p_sigmaInv,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        // one step for position
        double dt = std::max(std::min(p_lawSwitch(p_gen), p_T - p_timeInit), 1e-20);
        double sqrtdt = sqrt(dt);

        // store SDE solution
        Eigen::Matrix<double, N, 2 * P>  xProc;
        xProc.leftCols(P) = p_initVal;
        xProc.rightCols(P) = p_initVal; // antithetic

        // keep  gauss
        Eigen::Matrix< double, N, 1> gaussKeep;
        for (int id = 0; id < N; ++id)
            gaussKeep(id) = p_normal(p_gen);

        for (int i = 0; i < P ; ++i)
        {
            xProc.col(i) +=  p_mu * dt + p_sigma * sqrtdt * gaussKeep;
            xProc.col(i + P) +=  p_mu * dt - p_sigma * sqrtdt * gaussKeep;
        }
        // transpose annd multiply by gauss
        Eigen::Array<double, N, 1> sigInvGauss = (p_sigmaInv.transpose() * gaussKeep).array();
        // for return
        Eigen::Array< double, P, 1> ret ;
        Eigen::Array< double, N, P> retGrad;

        // new date
        double newDate = p_timeInit + dt;

        // test
        if ((newDate >=  p_T - 1e-5) || (p_depth == p_nbSim.size()))
        {
            double unSurCDF = 1. / (1. - p_lawSwitch.cdf(dt));
            for (int i = 0; i < P ; ++i)
            {
                double finalG = p_g(xProc.col(i));
                double finalGAnt = p_g(xProc.col(i + P));
                ret(i) = 0.5 * (finalG + finalGAnt) * unSurCDF;
                retGrad.col(i) = 0.5 * (finalG - finalGAnt) * unSurCDF / sqrtdt * sigInvGauss;
            }
        }
        else
        {
            int nbsimul = p_nbSim[p_depth];
            double lawSwitchPdf = p_lawSwitch.pdf(dt) ;
            Eigen::Array<double, 2 * P, 1 > yVal = Eigen::Array<double, 2 * P, 1 >::Zero();
            Eigen::Array<double, N, 2 * P>  zVal =  Eigen::Array<double, N, 2 * P >::Zero();
            // nesting recursion
            for (int is = 0 ; is < nbsimul; ++is)
            {
                std::pair< Eigen::Array<double, 2 * P, 1 >, Eigen::Array<double, N, 2 * P > >  solAndGrad =
                    SolveOneStepDyConst<N, 2 * P, SwitchDistrib, TRNGGenerator>()(xProc, p_mu, p_sigma, p_sigmaInv,  p_f, newDate, p_T, p_lawSwitch, p_normal,
                            p_gen, p_g,  p_nbSim,
                            p_depth + 1);
                yVal += solAndGrad.first;
                zVal += solAndGrad.second;
            }
            yVal /= nbsimul;
            zVal /= nbsimul;
            for (int i = 0; i < P; ++i)
            {
                double obstacle = p_f(newDate, xProc.col(i), yVal(i), zVal.col(i)) ;
                double obstacleAnt =  p_f(newDate, xProc.col(i + P), yVal(i + P), zVal.col(i + P));
                ret(i) = 0.5 * (obstacle + obstacleAnt) / lawSwitchPdf;
                retGrad.col(i) = 0.5 * (obstacle - obstacleAnt) / sqrtdt * sigInvGauss / lawSwitchPdf;
            }
        }
        return std::make_pair(ret, retGrad);
    }
};

/// \brief PDE resolution for  one step and one simulation with an Euler scheme
///        Suppose that the drift and diffusion are given by functions
//         Specialized template : permits to stop the template recursion
/// \param p_initVal           initial values for all points
/// \param p_mu                drift vector
/// \param p_sigma             diffusion matrix
/// \param p_sigmaInv          inverse of diffusion matrix
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value and gradient  at current date, current position
template< int N,   class SwitchDistrib, class TRNGGenerator>
class SolveOneStepDyConst<N, 64, SwitchDistrib, TRNGGenerator>
{
public:
    inline std::pair< Eigen::Array<double, 64, 1>, Eigen::Array< double, N, 64>  >   operator()(const Eigen::Matrix<double, N, 64>   &p_initVal,
            const Eigen::Matrix<double, N, 1>    &p_mu,
            const Eigen::Matrix<double, N, N>    &p_sigma,
            const Eigen::Matrix<double, N, N>    &p_sigmaInv,
            const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
            const double   &p_timeInit,
            const double &p_T,
            SwitchDistrib &p_lawSwitch,
            trng::normal_dist<double> &p_normal,
            TRNGGenerator &p_gen,
            const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
            const std::vector<int> &p_nbSim,
            int p_depth)
    {
        Eigen::Array<double, 64, 1> ret ;
        Eigen::Array< double, N, 64> retGrad;
        return std::make_pair(ret, retGrad);
    }
};


/// \brief PDE resolution for Euler Scheme
/// \param p_mu                drift vector
/// \param p_sigma             diffusion matrix
/// \param p_sigmaInv          inverse of diffusion matrix
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_point             initial point  where to calculate the solution
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_gen               parallel RNG
/// \return values std /sqrt(n) for val
template< int N, class SwitchDistrib, class TRNGGenerator >
std::tuple< double, Eigen::Array<double, N, 1>, double,  Eigen::Array<double, N, 1> >   solvePDEDYMCConst(const Eigen::Matrix<double, N, 1>    &p_mu,
        const Eigen::Matrix<double, N, N>    &p_sigma,
        const Eigen::Matrix<double, N, N>    &p_sigmaInv,
        const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &, const Eigen::Matrix<double, N, 1>&) > &p_f,
        const Eigen::Array<double, N, 1>   &p_point,
        const double   &p_timeInit,
        const double   &p_T,
        SwitchDistrib &p_lawSwitch,
        std::function< double (const Eigen::Matrix<double, N, 1>&)>   &p_g,
        const std::vector<int> &p_nbSim,
        TRNGGenerator &p_gen)
{

    // mpi
    boost::mpi::communicator world;
    //  random generator
    p_gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);

    // global simulation number
    int nbSimul = p_nbSim[0];
    int nbSimulProc = nbSimul / world.size();

    // store PDE value and std associated for first level
    double  val = 0.;
    double valSq =  0.;
    Eigen::Array<double, N, 1>  valGrad = Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>  valGradSq = Eigen::Array<double, N, 1>::Zero();
    int depth = 1;

    //#pragma omp parallel for schedule(dynamic,10)
    for (int is = 0; is <   nbSimulProc ; ++is)
    {
        if (world.rank() == 0)
            if (is % 1000 == 0)
                std::cout << " is " << is <<   std::endl ;
        std::pair<Eigen::Array<double, 1, 1>, Eigen::Array<double, N, 1> >  solLocAndGrad =  SolveOneStepDyConst<N, 1, SwitchDistrib, TRNGGenerator>()(p_point,  p_mu, p_sigma, p_sigmaInv, p_f, p_timeInit, p_T, p_lawSwitch, normal, p_gen, p_g, p_nbSim, depth);
        val += solLocAndGrad.first(0);
        valSq += solLocAndGrad.first(0) * solLocAndGrad.first(0);
        valGrad += solLocAndGrad.second;
        valGradSq += solLocAndGrad.second * solLocAndGrad.second;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, val, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, valSq, valSqRet, std::plus<double>());
    Eigen::Array<double, N, 1>  valGradRet =  Eigen::Array<double, N, 1>::Zero();
    Eigen::Array<double, N, 1>   valGradSqRet = Eigen::Array<double, N, 1>::Zero();
    boost::mpi::all_reduce(world, valGrad.data(), N, valGradRet.data(), std::plus<double>());
    boost::mpi::all_reduce(world, valGradSq.data(), N, valGradSqRet.data(), std::plus<double>());
    int nbSim = nbSimulProc * world.size();
    valRet /= nbSim;
    valSqRet /= nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / nbSim);
    valGradRet /= nbSim;
    valGradSqRet /= nbSim   ;
    valGradSqRet = ((valGradSqRet - valGradRet * valGradRet) / nbSim).sqrt();
    return std::make_tuple(valRet, valGradRet, valSqRet, valGradSqRet);
}
}

#endif
