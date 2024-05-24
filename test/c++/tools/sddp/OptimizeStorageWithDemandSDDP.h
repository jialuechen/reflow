
#ifndef OPTIMIZESTORAGEWITHDEMANDSDDP_H
#define OPTIMIZESTORAGEWITHDEMANDSDDP_H
#include "ClpSimplex.hpp"
#include <boost/lexical_cast.hpp>
#include "reflow/sddp/OptimizerSDDPBase.h"
#include "reflow/sddp/SDDPCutOptBase.h"
#include "test/c++/tools/sddp/OptimizeStorageWithDemandBaseSDDP.h"

/// \brief preprocessor helper
#define BClass  OptimizeStorageWithDemandBaseSDDP<Simulator>

/** \file OptimizeStorageWithDemandSDDP.h
 * \brief  Suppose that we have m_nbStorage Reservoir to satisfy demand
 *         Each reservoir is characterized by some inflows \f$f\f$ following an AR1 model
 *         \f$ f^{n+1} = \kappa (f^n -f)  + \sigma_f g  + \kappa f \f$ where \f$g\f$ is a Gaussian noise
 *         The goal is to satisfy a demand  following an AR1 model:
 *         \f$ D^{n+1} = k (D^n-D) + \sigma_d \tilde g + k D \f$ where \f$ \tilde g \f$ is a Gaussian noise
 *         When there is a shortage of water, energy is bought at a given price which is here deterministic
 * \author Xavier Warin
 */

/// \class AddConstraintStorage OptimizeStorageWithDemandSDDP.h
/// Add constraints to the Bellman value with cuts
class AddConstraintStorage
{

public:
    /// \brief  add constraints to Bellman value
    /// \param  p_linCut          cuts stored
    /// \param  p_nbStorage       number of storage
    /// \param  p_rows           rows for matrix contraints
    /// \param  p_columns        columns for matrix contraints
    /// \param  p_elements       A matrix elements
    /// \param  p_lowBoundConst  lower constraint \f$ lc\f$  on matrix \f$ lc \le A x \f$
    /// \param  p_upperBoundConst upper constraint \f$ uc\f$  on matrix \f$ A x \le uc \f$
    void addConstraints(const reflow::SDDPCutOptBase &p_linCut, int p_nbStorage,   Eigen::ArrayXi &p_rows,   Eigen::ArrayXi   &p_columns,  Eigen::ArrayXd   &p_elements,
                        Eigen::ArrayXd    &p_lowBoundConst,  Eigen::ArrayXd   &p_upperBoundConst) const
    {
        // get back cuts
        // here we use particle 0 because cuts are independent of a particle number
        Eigen::ArrayXd aParticle;
        Eigen::ArrayXXd  cuts =  p_linCut.getCutsAssociatedToAParticle(aParticle);
        int iBellPos = p_nbStorage * 3 + 2; // offset in variables
        int idecToStock = p_nbStorage; // p_nbStorage first values  used for withdrawal
        int isizeInit = p_elements.size();
        p_rows.conservativeResize(isizeInit + (2 * p_nbStorage + 2)*cuts.cols());
        p_columns.conservativeResize(isizeInit + (2 * p_nbStorage + 2)*cuts.cols());
        p_elements.conservativeResize(isizeInit + (2 * p_nbStorage + 2)*cuts.cols());
        int ibound = p_lowBoundConst.size();
        p_lowBoundConst.conservativeResize(ibound + cuts.cols());
        p_upperBoundConst.conservativeResize(ibound + cuts.cols());
        // add cuts
        for (int icut = 0 ; icut < cuts.cols() ; ++icut)
        {
            double affineValue = cuts(0, icut);
            int ipos = isizeInit + (2 * p_nbStorage + 2) * icut;
            p_rows(ipos) = ibound + icut;
            p_columns(ipos) = iBellPos;
            p_elements(ipos) = 1;
            for (int isto = 0 ; isto < p_nbStorage ; ++isto)
            {
                p_rows(ipos + isto + 1) = ibound + icut;
                p_columns(ipos + isto + 1) = idecToStock + isto;
                double derivStorage =  cuts(1 + isto, icut);
                p_elements(ipos + isto + 1) = -derivStorage;

                p_rows(p_nbStorage + ipos + isto + 1) = ibound + icut;
                p_columns(p_nbStorage + ipos + isto + 1) = idecToStock + p_nbStorage + isto;
                double derivInflows =  cuts(1 + isto + p_nbStorage, icut);
                p_elements(p_nbStorage + ipos + isto + 1) = - derivInflows;
            }
            // add demand
            double derivDemand =  cuts(1 + 2 * p_nbStorage, icut);
            p_rows(ipos + 2 * p_nbStorage + 1) = ibound + icut;
            p_columns(ipos + 2 * p_nbStorage + 1) = idecToStock + 2 * p_nbStorage;
            p_elements(ipos + 2 * p_nbStorage + 1) = -derivDemand;
            p_lowBoundConst(ibound + icut) = affineValue ;
            p_upperBoundConst(ibound + icut) = reflow::infty;
        }
    }
};


/// \class OptimizeStorageWithDemandSDDP OptimizeStorageWithDemandSDDP.h
///
template< class Simulator>
class OptimizeStorageWithDemandSDDP : public  BClass
{

private :


    std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> > m_timeSpot; ///< store the grid of spot price depending on time
    double m_spot ; ///< deterministic spot price
    double m_spotNext ; ///< deterministic spot price at next time step


public :


/// \brief Constructor for the storage problem
/// \param   p_withdrawalRate      withdrawal rate (volume) per time step
/// \param   p_nbStorage           Number of storage
/// \param   p_sigF                Volatility for inflows
/// \param   p_kappaF              AR coefficient for inflows
/// \param   p_timeInflowAver      Average inflow
/// \param   p_sigD                volatility for demand
/// \param   p_kappaD              AR coefficient for demand
/// \param   p_timeDAverage        average demand
/// \param   p_timeSpot            spot price
    /// \param   p_simulatorBackward   backward  simulator
    /// \param   p_simulatorForward    Forward simulator
    OptimizeStorageWithDemandSDDP(const double &p_withdrawalRate,  const int &p_nbStorage,
                                  const double &p_sigF,  const double &p_kappaF,
                                  const std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> >    &p_timeInflowAver,
                                  const  double   &p_sigD, const double &p_kappaD,
                                  const std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> > &p_timeDAverage,
                                  const  std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> >   &p_timeSpot,
                                  const std::shared_ptr<Simulator> &p_simulatorBackward,
                                  const std::shared_ptr<Simulator> &p_simulatorForward):
        BClass(p_withdrawalRate,  p_nbStorage, p_sigF,  p_kappaF, p_timeInflowAver, p_sigD,  p_kappaD,  p_timeDAverage,
               p_simulatorBackward, p_simulatorForward), m_timeSpot(p_timeSpot)
    {}



    /// \brief Optimize the LP during backward resolution
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_aState            Store the state, and 0, 0 values
    /// \param p_particle          Here no regression , so empty array
    /// \param p_isample            sample number for independent uncertainties
    /// \return  a vector with the optimal value and the derivatives if the function value with respect to each state (here the stocks)
    Eigen::ArrayXd oneStepBackward(const reflow::SDDPCutOptBase &p_linCut,
                                   const std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int > &p_aState,
                                   const Eigen::ArrayXd &p_particle, const int &p_isample) const
    {
        // constraints
        AddConstraintStorage constraints;
        // Creation  and PL resolution
        Eigen::ArrayXd stateFollowing(*std::get<0>(p_aState));
        // two cases : first date, take current uncertainty otherwise jump from previous to current state
        if (reflow::isLesserOrEqual(0., BClass::m_date))
        {
            // store new inflows and demand : cut such that it is positive
            for (int isto = 0; isto < BClass::m_nbStorage; ++isto)
                stateFollowing(isto + BClass::m_nbStorage) = std::max(BClass::m_kappaF * ((*std::get<0>(p_aState))(isto + BClass::m_nbStorage) - BClass::m_InflowAver) +  BClass::m_InflowAverNext + BClass::m_sigF * BClass::m_simulatorBackward->getGaussian(isto, p_isample), 0.);
            stateFollowing(2 * BClass::m_nbStorage) = std::max(BClass::m_kappaD * ((*std::get<0>(p_aState))(2 * BClass::m_nbStorage) - BClass::m_DAverage) + BClass::m_DAverageNext + BClass::m_sigD * BClass::m_simulatorBackward->getGaussian(BClass::m_nbStorage, p_isample), 0.);
        }
        Eigen::ArrayXd  valueAndDerivatives(2 + 2 * BClass::m_nbStorage);
        double cost ;
        BClass::createAndSolveLP(p_linCut, *std::get<0>(p_aState), constraints, m_spotNext, valueAndDerivatives, stateFollowing, cost);
        return valueAndDerivatives;
    }


    /// \brief Optimize the LP during forward resolution
    /// \param p_aParticle         a particle in simulation part to get back cuts
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_state             Store the state, the particle number used in optimization and mesh number associated to the particle. As an input it contains the current state
    /// \param p_stateToStore      For backward resolution we need to store \f$ (S_t,A_{t-1},D_{t-1}) \f$  where p_state in output is \f$ (S_t,A_{t},D_{t}) \f$
    /// \param p_isimu           number of the simulation used
    double  oneStepForward(const Eigen::ArrayXd &p_aParticle,  Eigen::ArrayXd &p_state,  Eigen::ArrayXd &p_stateToStore,
                           const reflow::SDDPCutOptBase &p_linCut,
                           const int &p_isimu) const
    {
        // optimizer constraints
        AddConstraintStorage constraints;
        // Creation  and PL resolution
        Eigen::ArrayXd stateFollowing(p_state);
        Eigen::ArrayXd  valueAndDerivatives(2 + 2 * BClass::m_nbStorage);
        double cost ;
        BClass::createAndSolveLP(p_linCut, p_state, constraints, m_spot, valueAndDerivatives, stateFollowing, cost);
        // store new inflows and demand
        for (int isto = 0; isto < BClass::m_nbStorage; ++isto)
            stateFollowing(isto + BClass::m_nbStorage) =  std::max(BClass::m_kappaF * (p_state(isto + BClass::m_nbStorage) - BClass::m_InflowAver) + BClass::m_InflowAverNext   + BClass::m_sigF * BClass::m_simulatorForward->getGaussian(isto, p_isimu), 0.);
        stateFollowing(2 * BClass::m_nbStorage) =  std::max(BClass::m_kappaD * (p_state(2 * BClass::m_nbStorage) - BClass::m_DAverage) + BClass::m_DAverageNext + BClass::m_sigD * BClass::m_simulatorForward->getGaussian(BClass::m_nbStorage, p_isimu), 0.);
        p_stateToStore  = p_state ;
        p_stateToStore.head(BClass::m_nbStorage) = stateFollowing.head(BClass::m_nbStorage); // only store stocks evolution
        p_state = stateFollowing;
        return cost;
    }

    /// brief update the optimizer for new dates (current date and next dates)
    void updateDates(const double &p_date, const double &p_dateNext)
    {
        BClass::updateDates(p_date, p_dateNext);
        if (reflow::isLesserOrEqual(0., p_date))
            m_spot = m_timeSpot->get(BClass::m_date);
        m_spotNext = m_timeSpot->get(BClass::m_dateNext);
    }


};
#endif
