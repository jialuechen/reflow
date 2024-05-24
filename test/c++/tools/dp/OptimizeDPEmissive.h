
#ifndef OPTIMIZEEMISSIVEREG_H
#define OPTIMIZEEMISSIVEREG_H
#include <Eigen/Dense>
#include <memory>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/regression/ContinuationValue.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/dp/OptimizerDPBase.h"
#include "test/c++/tools/simulators/AR1Simulator.h"


namespace reflow
{

/// \class OptimizeDPEmissive OptimizeDPEmissive.h
///  Class for optimize for Dynamic Programming for Non Emissive test case
class OptimizeDPEmissive: public reflow::OptimizerDPBase
{

    double m_alpha; // mean reverting coefficient for demand
    std::function< double(double, double)>   m_PI ;    // PI function depending on D and L the cumulative investment (gain for spot)
    std::function< double(double, double) >   m_cBar ;   // cbAR function depending on l (control) and L investment already achieved
    double m_s ;  // linear coefficient for subvention
    double m_lambda ; // coefficient for price of CO2
    double m_dt ; // time step
    double m_maturity ; // maturity
    double m_lMax ; // max of investment control
    double m_lStep ; // max for control
    std::vector <std::array< double, 2>  > m_extrem ;// extremal values of the grid
    std::shared_ptr< AR1Simulator> m_simulator ; // AR1 simulator for demand

public :

    /// \brief Constructor
    /// \param p_alpha     mean reverting coefficient for demand
    /// \param p_PI        gain from spot
    /// \param p_cBar      investment cost
    /// \param p_s         linear coefficient for subvention
    /// \param p_lambda    price CO2
    /// \param p_dt        resolution time step
    /// \param p_maturity  maturity of the control
    /// \param p_lMax      max of the investment control
    /// \param p_lStep     Step discretization for l
    /// \param p_extrem   extremal point of the grid
    OptimizeDPEmissive(const double &p_alpha,
                       const std::function<double(double, double)> &p_PI,
                       const std::function< double(double, double) >     &p_cBar,  const double   &p_s, const double &p_lambda,
                       const double &p_dt,
                       const double &p_maturity,
                       const  double &p_lMax, const double &p_lStep, const  std::vector <std::array< double, 2>  >   &p_extrem);

    virtual ~OptimizeDPEmissive() {}

    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    virtual std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const;


    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const;


    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinate of the stock point to treat
    /// \param p_condEsp   continuation values operator for each regime
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> stepOptimize(const   std::shared_ptr< reflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
            const std::vector< ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn)  const;


    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
    /// This implementation is for test and example purpose
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified)
    virtual void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridAndRegressedValue > &p_continuation,
                              reflow::StateWithStocks &p_state,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const;


    /// \brief Defines a step in simulation using interpolation in controls
    void stepSimulateControl(const std::shared_ptr< reflow::SpaceGrid> &, const std::vector< reflow::GridAndRegressedValue  > &,
                             reflow::StateWithStocks &, Eigen::Ref<Eigen::ArrayXd>) const {}

    /// \brief Get the number of regimes allowed for the asset to be reached  at the current time step
    ///    If \f$ t \f$ is the current time, and $\f$ dt \f$  the resolution step,  this is the number of regime allowed on \f$[ t- dt, t[\f$
    virtual   int getNbRegime() const
    {
        return 2;
    }

    /// \brief get the simulator back
    virtual std::shared_ptr< reflow::SimulatorDPBase > getSimulator() const
    {
        return m_simulator;
    }

    /// \brief  select a simulator
    /// \param p_simulator simulator to use
    inline void setSimulator(const std::shared_ptr<AR1Simulator> &p_simulator)
    {
        m_simulator =  p_simulator ;
    }

    /// \brief get size of the  function to follow in simulation
    ///        here we follow the function value
    inline int getSimuFuncSize() const
    {
        return 2 ;
    }
    // \brief get back the dimension of the control
    inline int getNbControl() const
    {
        return 1 ;
    }

};
}
#endif /* OPTIMIZEEMISSIVEBASE_H */
