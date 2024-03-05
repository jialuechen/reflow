
#ifndef OPTIMIZEOPTIONL2_H
#define OPTIMIZEOPTIONL2_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/utils/constant.h"
#include "libflow/regression/LocalRegression.h"
#include "libflow/regression/globalL2HedgeMinimize.h"

/* \file OptimizeOptionL2.h
 * \brief Permits to optimize and simulate in one step at one stock level (in hedging product)  in libflow  framework
 *        an option with a global variance minimization.
 * \author Xavier Warin
 */
template< class Simulator>
class OptimizeOptionL2: public libflow::OptimizerDPBase //, public libflow::OptimizerNoRegressionDPBase
{
private :

    Eigen::ArrayXd  m_posVarBuy ; ///< Max variation position  buy (>0)
    Eigen::ArrayXd  m_posVarSell ; ///< Max variation position sell (>0)
    Eigen::ArrayXd  m_stepForHedge ; ///< Step for command
    Eigen::ArrayXd  m_spread1 ; ///< Constant part of half of spread bid asset
    Eigen::ArrayXd  m_spread2 ; ///< Linear part of half of spread bid ask
    /// \brief store the simulator
    std::shared_ptr<Simulator> m_simulator;
    Eigen::ArrayXXd m_assetVar  ; /// \brief Asset values variations between two time step with optimization (nb_asset, nb_simu)
    Eigen::ArrayXXd m_asset ; ///\brief  (nb_asset, nb_simu)

public :

    /// \param p_posVarBuy     Max variation position  buy (>0) per asset
    /// \param m_posVarSell    Max variation position sell (>0) per asset
    /// \param p_stepForHedge  step for commands per asset
    /// \param p_spread1       constant part of semi spread bid ask per asset
    /// \param p_spread2       linear part of semi spread bid ask per asset
    OptimizeOptionL2(const   Eigen::ArrayXd   &p_posVarBuy,
                     const   Eigen::ArrayXd   &p_posVarSell,
                     const   Eigen::ArrayXd   &p_stepForHedge,
                     const   Eigen::ArrayXd   &p_spread1,
                     const   Eigen::ArrayXd   &p_spread2):
        m_posVarBuy(p_posVarBuy), m_posVarSell(p_posVarSell),
        m_stepForHedge(p_stepForHedge), m_spread1(p_spread1), m_spread2(p_spread2)
    {
    }

    /// \brief define the diffusion cone for parallelism
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector< std::array< double, 2>  > &) const
    {
        std::vector< std::array< double, 2> > extrGrid(m_posVarBuy.size());
        for (int i = 0 ; i <  m_posVarBuy.size(); ++i)
        {
            extrGrid[i][0] =  -libflow::infty; // get all Bellmna Values independently  of the position
            extrGrid[i][1] =   libflow::infty; // get all Beelman values independently  of the position
        }
        return extrGrid;
    }

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(m_posVarBuy.size(), true);
        return  bDim ;
    }

    /// \brief permit  store modification in asset values
    /// \param  p_assetVar   asset variation between two optimisation
    /// \param  p_asset      asset value at current time step
    void setCommandsAndAssetVar(const Eigen::ArrayXXd &p_assetVar, const Eigen::ArrayXXd &p_asset)
    {
        m_assetVar = p_assetVar;
        m_asset = p_asset;
    }

    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinate of the stock point to treat
    /// \param p_condEsp   continuation values for each regime
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///
    std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> stepOptimize(const std::shared_ptr< libflow::SpaceGrid> &p_grid,
            const Eigen::ArrayXd   &p_stock,
            const std::vector<libflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        int nbSimul = m_simulator->getNbSimul();
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        solutionAndControl.first.resize(nbSimul, 1);
        solutionAndControl.second.resize(nbSimul, 1);
        // update commands and step commands in order to catch Bang Bang constrol
        Eigen::ArrayXd posMin = p_stock ;
        Eigen::ArrayXd stepCommand(posMin.size()) ;
        Eigen::ArrayXi nbCommand(posMin.size());
        for (int id = 0; id < posMin.size(); ++id)
        {
            double posMax = p_stock(id);
            posMin(id) = std::max(p_stock(id) - m_posVarSell(id), p_grid->getExtremeValues()[id][0]);
            posMax = std::min(p_stock(0) + m_posVarBuy(id), p_grid->getExtremeValues()[id][1]);
            nbCommand(id) = static_cast<int>((posMax - posMin(id) + libflow::tiny) / m_stepForHedge(id));
            stepCommand(id) = (posMax - posMin(id)) / nbCommand(id);
        }
        // command grid
        std::shared_ptr<libflow::SpaceGrid>  commands = std::make_shared<libflow::RegularSpaceGrid>(posMin, stepCommand, nbCommand);
        std::pair<Eigen::ArrayXd, Eigen::ArrayXXd> valueAndHedge;
        std::shared_ptr< libflow::BaseRegression >  regressor = p_condEsp[0].getCondExp();
        valueAndHedge = globalL2HedgeMinimize(m_assetVar, m_asset, p_stock, m_spread1, m_spread2, *commands, regressor, *p_grid, *p_phiIn[0]);
        // store solution
        solutionAndControl.first.col(0) = valueAndHedge.first;
        solutionAndControl.second.col(0) = valueAndHedge.second.col(0);
        return solutionAndControl;
    }

    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid>    &p_grid,
                             const std::vector< libflow::GridAndRegressedValue  > &p_control,
                             libflow::StateWithStocks &p_state,
                             Eigen::Ref<Eigen::ArrayXd> p_phiInOut)  const
    {
        // previsous delta
        Eigen::ArrayXd ptStock = p_state.getPtStock();
        // optimal control : new delta for each stock
        Eigen::ArrayXd pControl(ptStock.size());
        for (int iCont = 0; iCont < pControl.size(); ++iCont)
        {
            pControl(iCont) = p_control[iCont].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
            // cap the control to respect the constraints
            double incStock = std::min(std::max(pControl(iCont) - ptStock(iCont), -m_posVarSell(iCont)), m_posVarBuy(iCont));
            pControl(iCont) = ptStock(iCont) + incStock;
            // local global after
            pControl(iCont) = std::min(std::max(p_grid->getExtremeValues()[iCont][0], pControl(iCont)), p_grid->getExtremeValues()[iCont][1]);
        }
        // add spread
        p_phiInOut(0) += ((pControl - ptStock).abs() * (m_spread1 + m_spread2 * p_state.getStochasticRealization())).sum() ;
        // store new position in stock (modification due to asset achieve outside
        p_state.setPtStock(pControl);
    }

    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal but is convenient if the control is discrete.
    /// By avoiding interpolation in control we avoid non admissible control
    /// Control are recalculated during simulation.
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    void stepSimulate(const std::shared_ptr< libflow::SpaceGrid> &,
                      const std::vector< libflow::GridAndRegressedValue  > &,
                      libflow::StateWithStocks &,
                      Eigen::Ref<Eigen::ArrayXd>) const {}

    /// \brief get number of regimes
    inline int getNbRegime() const
    {
        return 1 ;
    }

    /// \brief get number of regime reached at the end of the resolution
    inline int getNbRegimeReached() const
    {
        return 1;
    }

    /// \brief number of controls
    inline int getNbControl() const
    {
        return m_posVarBuy.size();
    }


    ///\brief store the simulator
    virtual inline void setSimulator(const std::shared_ptr<Simulator> &p_simulator)
    {
        m_simulator = p_simulator ;
    }

    /// \brief get the simulator back
    inline std::shared_ptr< libflow::SimulatorDPBase > getSimulator() const
    {
        return m_simulator ;
    }

    /// \brief get size of the  function to follow in simulation
    inline int getSimuFuncSize() const
    {
        return 1 ;
    }

    /// \brief Get back constant part of spread
    inline Eigen::ArrayXd getConstSpread() const
    {
        return m_spread1;
    }

    /// \brief Get back linear part of the spread
    inline Eigen::ArrayXd getLinSpread() const
    {
        return m_spread2;
    }
};

#endif
