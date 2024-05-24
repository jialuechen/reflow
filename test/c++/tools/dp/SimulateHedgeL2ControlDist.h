
#ifndef  SIMULATEHEDGEL2CONTROLDIST_H
#define  SIMULATEHEDGEL2CONTROLDIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/dp/SimulateStepRegressionControlDist.h"
#include "OptimizeOptionL2.h"
#include "reflow/dp/SimulatorDPBase.h"


template< class PriceModel>
double SimulateHedgeL2ControlDist(const std::shared_ptr<reflow::FullGrid> &p_grid,
                                  const std::shared_ptr<OptimizeOptionL2<PriceModel> > &p_optimize,
                                  const std::function<double(const Eigen::ArrayXd &)>  &p_funcFinalValue,
                                  const double   &p_optionValue,
                                  const std::string   &p_fileToDump,
                                  const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    std::shared_ptr< reflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< reflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    // to store previous asset value
    std::shared_ptr<Eigen::ArrayXXd> assetPrev = std::make_shared<Eigen::ArrayXXd>(simulator->getParticles());

    // initial state (no hedge in portfolio  and  initial asset value)
    Eigen::ArrayXd  initAssetValue = simulator->getParticles().col(0).transpose();
    reflow::StateWithStocks initState(0, Eigen::ArrayXd::Zero(simulator->getDimension()), initAssetValue);
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(initState);
    std::string toDump = p_fileToDump;
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Storage";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    // cost of trading
    Eigen::ArrayXd  costTrading = Eigen::ArrayXd::Zero(simulator->getNbSimul());


    for (int istep = 0; istep < nbStep; ++istep)
    {
        if (p_world.rank() == 0)
            std::cout << "Step " << istep << " out of " << nbStep << std::endl ;

        reflow::SimulateStepRegressionControlDist(ar, istep, nameAr, p_grid, p_grid, std::static_pointer_cast<reflow::OptimizerDPBase>(p_optimize), true, p_world).oneStep(states, costFunction);

        // new stochastic state
        std::shared_ptr<Eigen::ArrayXXd> assetNext = std::make_shared<Eigen::ArrayXXd>(simulator->stepForwardAndGetParticles());
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(assetNext->col(is));
        // cost evolution due to trading
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            for (int iAsset = 0; iAsset < assetPrev->rows(); ++iAsset)
                costTrading(is) -= ((*assetNext)(iAsset, is) - (*assetPrev)(iAsset, is)) * states[is].getPtOneStock(iAsset);
        // shift
        assetPrev = assetNext;
    }

    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    Eigen::ArrayXd payOff = Eigen::ArrayXd::Zero(simulator->getNbSimul());
    Eigen::ArrayXd constSpread = p_optimize->getConstSpread();
    Eigen::ArrayXd linSpread = p_optimize->getLinSpread();
    Eigen::ArrayXd toStoreStock(assetPrev->rows());
    Eigen::ArrayXd toStoreAsset(assetPrev->rows());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
    {
        // pay off + final liquidation
        payOff(is) += p_funcFinalValue(states[is].getStochasticRealization()) ;
        // state
        toStoreStock = states[is].getPtStock();
        toStoreAsset = states[is].getStochasticRealization();
        // cost
        for (int iAsset = 0; iAsset < assetPrev->rows(); ++iAsset)
            costFunction(0, is) += std::fabs(toStoreStock(iAsset)) * (constSpread(iAsset) + toStoreAsset(iAsset) * linSpread(iAsset));
    }

    // for RETURN
    Eigen::ArrayXd ret = Eigen::ArrayXd::Zero(simulator->getNbSimul());
    ret += costFunction.row(0).transpose();
    ret += costTrading;
    ret += payOff;

    if (p_world.rank() == 0)
    {
        double meanRet = ret.mean();
        double stdRet = sqrt((ret * ret).mean() - meanRet * meanRet);
        std::cout << " Prod  Val simul " <<  ret.mean() <<  "Std Ret " << stdRet << std::endl ;
        std::cout << "########################################################################" << std::endl;
    }
    return ret.mean();
}

#endif
