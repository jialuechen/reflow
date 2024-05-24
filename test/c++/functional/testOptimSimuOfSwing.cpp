
#define BOOST_TEST_DYN_LINK
#include <functional>
#include <boost/mpi.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <Eigen/Dense>
#include "reflow/core/utils/types.h"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/utils/InterpolatorEncapsulated.h"
#include "reflow/core/utils/GridIterator.h"
#include "reflow/regression/LocalLinearRegression.h"
#include "reflow/regression/ContinuationValue.h"
#include "test/c++/tools/EuropeanOptions.h"
#include "test/c++/tools/BlackScholesSimulator.h"
#include "test/c++/tools/DynamicProgrammingByRegression.h"

double accuracyClose = 0.2;
double accuracyEqual = 1e-10;

using namespace std;
using namespace Eigen ;


/// \brief  call payoff for basket
class BasketCall
{
    double m_strike;
public:
    explicit BasketCall(const double &p_strike): m_strike(p_strike) {}

    // For one simulation
    inline double apply(const VectorXd &assets) const
    {
        return std::max(assets.mean() - m_strike, 0.);
    }
    // for all simulations
    VectorXd applyVec(const MatrixXd &assets) const
    {
        VectorXd ret(assets.cols());
        for (int is = 0 ; is < assets.cols(); ++is)
        {
            ret(is) = std::max(assets.col(is).mean() - m_strike, 0.);
        }
        return ret;
    }
};


/// \brief final function payoff
template< class PayOff >
class FinalValueFunction
{
private :

    PayOff m_pay ;

public :
    /// \brief Constructor
    explicit FinalValueFunction(const PayOff &p_pay): m_pay(p_pay) {}


/// \brief final function
/// \param  p_stock  position in the stock
/// \param  p_state  position in the stochastic state
    inline double operator()(const Eigen::ArrayXd &p_stock, const Eigen::ArrayXd   &p_state) const
    {
        return  m_pay.apply(p_state.matrix());
    }
};


/// \class OptimizerSwing testSwing.cpp
template< class PayOff>
class OptimizeSwing
{
private :

    PayOff m_payoff;///< pay off function
    int m_nPointStock ; ///< number of point stocks
    double m_actu ; /// actualisation per step

public :

    /// \brief Constructor
    /// \param p_payoff pay off used
    /// \param  p_nPointStock number of stock points
    /// \param p_actu   actualization factor
    OptimizeSwing(const PayOff &p_payoff, const int &p_nPointStock, const double p_actu): m_payoff(p_payoff), m_nPointStock(p_nPointStock), m_actu(p_actu) {}

    /// \brief define the diffusion cone for parallelism
    /// \param region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< boost::array< double, 2> > getCone(const  std::vector<  boost::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< boost::array< double, 2> > extrGrid(p_regionByProcessor);
        // only a single  exercise
        extrGrid[0][1] += 1.;
        return extrGrid;
    }

/// \brief defines a step in optimization
/// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
/// This implementation is for test and example purpose
/// \param p_stock     coordinate of the stock point to treat
/// \param p_condEsp   conditional expectation operator
/// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
/// \param p_interpol  Defines the interpolator for stock at the previous step
/// \return for each regimes (column) gives the solution for each particle (row)
    template< class CondEspectation, class Grid >
    Eigen::ArrayXXd step(const ArrayXd   &p_stock, const CondEspectation &p_condEsp,  const std::vector < boost::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                         const  InterpolatorEncapsulated< LinearInterpolator< Grid >, Grid>   &p_interpol) const
    {
        ArrayXXd solution(p_condEsp.getParticles().cols(), 1);
        ArrayXd payOffVal = m_payoff.applyVec(p_condEsp.getParticles()).array();
        // calculation detailed for clarity
        // create interpolator at current stock point
        LinearInterpolator< Grid >  interpolatorCurrentStock = p_interpol.createInterpolator(p_stock);
        // cash flow at current stock and previous step
        ArrayXd cashSameStock = interpolatorCurrentStock.applyVec(*p_phiIn[0]);
        // conditional expectation at current stock point
        ArrayXd condExpSameStock = p_condEsp.getAllSimulations(m_actu * cashSameStock.matrix()).array();
        if (p_stock(0) < m_nPointStock - 1)
        {
            // calculation detailed for clarity
            // create interpolator at next stock point accessible
            ArrayXd nextStock(p_stock);
            nextStock(0) += 1;
            LinearInterpolator< Grid >  interpolatorNextStock = p_interpol.createInterpolator(nextStock);
            // cash flow at next stock previous step
            ArrayXd cashNextStock = interpolatorNextStock.applyVec(*p_phiIn[0]);
            // conditional espectation at next stock
            ArrayXd condExpNextStock = p_condEsp.getAllSimulations(m_actu * cashNextStock.matrix()).array();
            // arbitrage
            solution.col(0) = (payOffVal + condExpNextStock  > condExpSameStock).select(payOffVal + m_actu * cashNextStock, m_actu * cashSameStock);
        }
        else
        {
            // arbitrage
            solution.col(0) = (payOffVal > condExpSameStock).select(payOffVal, m_actu * cashSameStock);

        }
        return solution;
    }
/// \brief get number of regimes
    inline int getNbRegime() const
    {
        return 1;
    }
}
;

/// \brief Analytical value
///  \param p_S        asset value
///  \param p_sigma   volatility
///  \param p_r       interest rate
///  \param p_strike  strike
///  \param p_dates     possible exercise dates
///  \return option value
double analyticalValue(const int N, const double &p_S, const double &p_sigma, const double &p_r, const double   &p_strike,
                       const Eigen::ArrayXd &p_dates)
{
    double analytical = 0.;
    for (int i = p_dates.size() - N; i < p_dates.size(); ++i)
        analytical += CallOption()(p_S, p_sigma,  p_r, p_strike,  p_dates(i));
    return analytical;
}


/// \brief Classical resolution for swing
/// \param p_sim      Monte Carlo simulator
/// \param p_payOff   Option pay off
/// \param p_nbMesh   Meshing
/// \param p_dates    possible exercise dates
/// \param p_N          number of exercises
template < class Simulator, class PayOff  >
double resolutionSwingLocalRegression(Simulator &p_sim, const PayOff &p_payOff, const ArrayXi   &p_nbMesh, const Eigen::ArrayXd &p_dates, const int &p_N)
{
    assert((p_sim.getNbStep() + 1) == p_dates.size());

    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    // in this example the step between two exercises is given
    double expRate = exp(-p_sim.getStep() * p_sim.getMu()(0));

    // final payoff
    VectorXd finalPayOff(p_payOff.applyVec(p_sim.getAssetValues()));
    // Terminal function depending on simulations and stock already exercised
    boost::shared_ptr<MatrixXd> cashNext = boost::make_shared<MatrixXd>(finalPayOff.size(), p_N);
    for (int is = 0; is < finalPayOff.size(); ++is)
        cashNext->row(is) = Eigen::VectorXd::Constant(p_N, finalPayOff(is)).transpose();
    boost::shared_ptr<MatrixXd> cashPrev = boost::make_shared<MatrixXd>(finalPayOff.size(), p_N);
    for (int iStep = p_dates.size() - 2; iStep >= 0; --iStep)
    {
        MatrixXd asset(p_sim.stepBackward());
        VectorXd payOffLoc = p_payOff.applyVec(asset);
        // conditional expectation
        LocalLinearRegression regressor(((iStep == 0) ? true : false), asset, p_nbMesh);

        // store conditional expectations
        std::vector< boost::shared_ptr< VectorXd > > vecCondEspec(p_N);
        for (int iStock = 0 ; iStock < p_N; ++iStock)
            vecCondEspec[iStock] = boost::make_shared<VectorXd>(regressor.getAllSimulations(cashNext->col(iStock)) * expRate);

        // arbitrage
        for (int iStock = 0 ; iStock < p_N - 1; ++iStock)
            cashPrev->col(iStock) = (payOffLoc.array() + vecCondEspec[iStock + 1]->array() >  vecCondEspec[iStock]->array()).select(payOffLoc + expRate * cashNext->col(iStock + 1),
                                    expRate * cashNext->col(iStock));
        // last stock
        cashPrev->col(p_N - 1) = (payOffLoc.array() >  vecCondEspec[p_N - 1]->array()).select(payOffLoc, expRate * cashNext->col(p_N - 1));

        // switch pointer
        boost::shared_ptr<MatrixXd> tempVec = cashNext;
        cashNext = cashPrev;
        cashPrev = tempVec;
    }
    return cashNext->col(0).mean();
}


/// \brief Same resolution using  Continuation Object to deal with stocks
/// \param p_sim      Monte Carlo simulator
/// \param p_payOff   Option pay off
/// \param p_nbMesh   Meshing
/// \param p_dates    possible exercise dates
/// \param p_N        number of exercises
/// \param p_file     file to dump in
template < class Simulator, class PayOff  >
double resolutionSwingContinuationLocalRegression(Simulator &p_sim, const PayOff &p_payOff, const ArrayXi   &p_nbMesh, const Eigen::ArrayXd &p_dates, const int &p_N,
        std::string p_file)
{
    assert((p_sim.getNbStep() + 1) == p_dates.size());
    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    // in this example the step between two exercises is given
    double expRate = exp(-p_sim.getStep() * p_sim.getMu()(0));
    // regular grid
    Eigen::ArrayXd lowValues(1), step(1);
    lowValues(0) = 0. ;
    step(0) = 1;
    Eigen::ArrayXi  nbStep(1);
    nbStep(0) = p_N - 1;
    boost::shared_ptr< RegularSpaceGrid > regular = boost::make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // extremal values of the grid
    std::vector <boost::array< double, 2>  > extremeGrid = regular->getExtremeValues();

    // final payoff
    VectorXd finalPayOff(p_payOff.applyVec(p_sim.getAssetValues()));
    // Terminal function depending on simulations and stock already exercised
    boost::shared_ptr<ArrayXXd> cashNext = boost::make_shared<ArrayXXd>(finalPayOff.size(), regular->getNbPoints());
    for (int is = 0; is < finalPayOff.size(); ++is)
        cashNext->row(is) = Eigen::ArrayXd::Constant(regular->getNbPoints(), finalPayOff(is)).transpose();
    boost::shared_ptr<ArrayXXd> cashPrev = boost::make_shared<ArrayXXd>(finalPayOff.size(), regular->getNbPoints());
    for (int iStep = p_dates.size() - 2; iStep >= 0; --iStep)
    {
        ArrayXXd asset(p_sim.stepBackward());
        ArrayXd payOffLoc = p_payOff.applyVec(asset).array();
        // conditional expectation
        boost::shared_ptr<LocalLinearRegression> regressor = boost::make_shared<LocalLinearRegression>(((iStep == 0) ? true : false), asset, p_nbMesh);
        // continuation value object dealing with stocks
        ContinuationValue<RegularSpaceGrid, LocalLinearRegression> continuation(regular, regressor, *cashNext);
        // iterator on grid points
        GridIterator<RegularSpaceGrid> iterOnGrid(*regular);
        while (iterOnGrid.isValid())
        {
            ArrayXd CoordStock = regular->getCoordinateFromIntCoord(iterOnGrid.get());
            // use continuation to get realization of condition expectation
            ArrayXd conditionExpecCur = expRate * continuation.getAllSimulations< LinearInterpolator< RegularSpaceGrid> >(CoordStock);
            if (isLesserOrEqual(CoordStock(0) + 1, extremeGrid[0][1]))
            {
                ArrayXd conditionExpecNext = expRate * continuation.getAllSimulations< LinearInterpolator< RegularSpaceGrid> >(CoordStock + 1);
                cashPrev->col(iterOnGrid.getCount()) = (payOffLoc + conditionExpecNext >  conditionExpecCur).select(payOffLoc + expRate * cashNext->col(iterOnGrid.getCount() + 1),
                                                       expRate * cashNext->col(iterOnGrid.getCount()));
            }
            else
            {
                cashPrev->col(iterOnGrid.getCount()) = (payOffLoc >  conditionExpecCur).select(payOffLoc,	expRate * cashNext->col(iterOnGrid.getCount()));
            }
            iterOnGrid.next();
        }
        // switch pointer
        boost::shared_ptr<ArrayXXd> tempVec = cashNext;
        cashNext = cashPrev;
        cashPrev = tempVec;
    }
    return cashNext->col(0).mean();
}




BOOST_AUTO_TEST_CASE(testSwingOptionInOptimization)
{
    boost::mpi::communicator world;
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 100;
    // exercise date
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., 1.);
    int N = 10 ; // 10 exercise dates
    double T = 1. ;
    double strike = 1.;
    int nbSimul = 200000;
    int nMesh = 16;
    // payoff
    BasketCall  payoff(strike);
    // analytical
    double analytical = ((world.rank() == 0) ? analyticalValue(N, initialValues(0), sigma(0), mu(0), strike, dates) : 0);
    // store sequential
    double valueSeq ;
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    if (world.rank() == 0)
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // Bermudean value
        valueSeq = resolutionSwingLocalRegression(simulator, payoff, nbMesh, dates, N);
        BOOST_CHECK_CLOSE(valueSeq, analytical, accuracyClose);
    }
    if (world.rank() == 0)
    {
        double  valueSeqContinuation ;
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // using continuation values
        valueSeqContinuation = resolutionSwingContinuationLocalRegression(simulator, payoff, nbMesh, dates, N, "ToDumpSwing");
        BOOST_CHECK_EQUAL(valueSeq, valueSeqContinuation);
    }
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, 1.);
    ArrayXi nbStep = ArrayXi::Constant(1, N - 1);
    boost::shared_ptr<RegularSpaceGrid> grid = boost::make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    std::vector< boost::shared_ptr<std::function<double(const Eigen::ArrayXd &, const Eigen::ArrayXd &)> > > vFunction(1);
    vFunction[0] = boost::make_shared< std::function<double(const Eigen::ArrayXd &, const Eigen::ArrayXd &)> >(FinalValueFunction<BasketCall>(payoff));
    // optimizer
    double actu = exp(-mu(0) * T / nStep);
    boost::shared_ptr< OptimizeSwing<BasketCall> >optimizer = boost::make_shared< OptimizeSwing<BasketCall> >(payoff, N, actu);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, 0.);
    int initialRegime = 0;
    std::string fileToDump = "CondExp";
    bool bOneFile = false;
    boost::tuple<bool, double > valueParal = DynamicProgrammingByRegression(simulator, grid, optimizer, nbMesh, vFunction, initialStock, initialRegime, fileToDump, bOneFile);
    if (valueParal.get<0>())
    {
        BOOST_CHECK_EQUAL(valueSeq, valueParal.get<1>());
    }

    std::string fileToDumpThread = "CondExpThread";
    if (world.rank() == 0)
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // using continuation values
        double valueThread =  DynamicProgrammingByRegressionThread(simulator, grid, optimizer, nbMesh, vFunction, initialStock, initialRegime, fileToDumpThread);
        std::cout << " valueSeq " << valueSeq << " valueThread " << valueThread << std::endl ;
        BOOST_CHECK_EQUAL(valueSeq, valueThread);
    }

}





/// \class OptimizerFictitiousSwing testSwing.cpp
///  Defines a fictitious swing on ndim stocks : the valorisation will be equal to the valorisation of ndim swing
///  where ndim is the number of stocks
template< class PayOff>
class OptimizeFictitiousSwing
{
private :

    PayOff m_payoff;///< pay off function
    int m_nPointStock ; ///< number of point stocks per dimension
    int m_ndim ; ///< number of stocks to deal with
    double m_actu ; /// actualisation per step

public :

    /// \brief Constructor
    /// \param p_payoff pay off used
    /// \param  p_nPointStock number of stock points
    /// \param p_actu   actualization factor
    OptimizeFictitiousSwing(const PayOff &p_payoff, const int &p_nPointStock, const int &p_ndim, const double &p_actu): m_payoff(p_payoff), m_nPointStock(p_nPointStock),
        m_ndim(p_ndim), m_actu(p_actu) {}

    /// \brief define the diffusion cone for parallelism
    /// \param region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< boost::array< double, 2> > getCone(const  std::vector<  boost::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< boost::array< double, 2> > extrGrid(p_regionByProcessor);
        // only a single  exercise
        for (int id = 0; id < m_ndim; ++id)
            extrGrid[id][1] += 1.;
        return extrGrid;
    }

/// \brief defines a step in optimization
/// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
/// This implementation is for test and example purpose
/// \param p_stock     coordinate of the stock point to treat
/// \param p_condEsp   conditional expectation operator
/// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
/// \param p_interpol  Defines the interpolator for stock at the previous step
/// \return for each regimes (column) gives the solution for each particle (row)
    template< class CondEspectation, class Grid >
    Eigen::ArrayXXd step(const ArrayXd   &p_stock, const CondEspectation &p_condEsp,  const std::vector < boost::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                         const  InterpolatorEncapsulated< LinearInterpolator< Grid >, Grid>   &p_interpol) const
    {
        ArrayXXd solution(p_condEsp.getParticles().cols(), 1);
        ArrayXd payOffVal = m_payoff.applyVec(p_condEsp.getParticles()).array();
        // calculation detailed for clarity
        // create interpolator at current stock point
        LinearInterpolator< Grid >  interpolatorCurrentStock = p_interpol.createInterpolator(p_stock);
        // cash flow at current stock and previous step
        ArrayXd cashSameStock = interpolatorCurrentStock.applyVec(p_phiIn[0]->array());
        // conditional expectation at current stock point
        ArrayXd condExpSameStock = p_condEsp.getAllSimulations(m_actu * cashSameStock.matrix()).array();
        // number of possibilities for arbitrage with m_ndim stocks
        int nbArb = (0x01 << m_ndim);
        // stock to add
        ArrayXd stockToAdd(m_ndim);
        for (int j = 0 ; j < nbArb; ++j)
        {
            unsigned int ires = j ;
            for (int id = m_ndim - 1 ; id >= 0  ; --id)
            {
                unsigned int idec = (ires >> id) ;
                stockToAdd(id) = idec;
                ires -= (idec << id);
            }
            // calculation detailed for clarity
            // create interapolator at next stock point accessible
            ArrayXd nextStock = p_stock + stockToAdd;
            LinearInterpolator< Grid >  interpolatorNextStock = p_interpol.createInterpolator(nextStock);
            // cash flow at next stock previous step
            ArrayXd cashNextStock = interpolatorNextStock.applyVec(p_phiIn[0]->array());
            // conditional espectation at next stock
            ArrayXd condExpNextStock = p_condEsp.getAllSimulations(m_actu * cashNextStock.matrix()).array();
            // arbitrage
            solution.col(0) = (payOffVal + condExpNextStock  > condExpSameStock).select(payOffVal + m_actu * cashNextStock, m_actu * cashSameStock);
        }
        else
        {
            // arbitrage
            solution.col(0) = (payOffVal > condExpSameStock).select(payOffVal, m_actu * cashSameStock);

        }
        return solution;
    }
/// \brief get number of regimes
    inline int getNbRegime() const
    {
        return 1;
    }
}
;


/// \brief function to test stock in dimension above 1
int testMultiStock(const int p_ndim)
{
    boost::mpi::communicator world;
    VectorXd initialValues = ArrayXd::Constant(p_ndim, 1.);
    VectorXd sigma  = ArrayXd::Constant(p_ndim, 0.2);
    VectorXd mu  = ArrayXd::Constant(p_ndim, 0.05);
    MatrixXd corr = MatrixXd::Zero(p_ndim, p_ndim);
    cor.diagonal().setConstant(1.);
    // number of step
    int nStep = 100;
    // exercise date
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., 1.);
    int N = 10 ; // 10 exercise dates
    double T = 1. ;
    double strike = 1.;
    int nbSimul = 10;
    int nMesh = 16;
    // payoff
    BasketCall  payoff(strike);
    // store sequential
    double valueSeq ;
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    if (world.rank() == 0)
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // Bermudean value
        valueSeq = resolutionSwingLocalRegression(simulator, payoff, nbMesh, dates, N);
    }
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
    // grid
    ArrayXd lowValues = ArrayXd::Constant(p_ndim, 0.);
    ArrayXd step = ArrayXd::Constant(p_ndim, 1.);
    ArrayXi nbStep = ArrayXi::Constant(p_ndim, N - 1);
    boost::shared_ptr<RegularSpaceGrid> grid = boost::make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    std::vector< boost::shared_ptr<std::function<double(const Eigen::ArrayXd &, const Eigen::ArrayXd &)> > > vFunction(1);
    vFunction[0] = boost::make_shared< std::function<double(const Eigen::ArrayXd &, const Eigen::ArrayXd &)> >(FinalValueFunction<BasketCall>(payoff));
    // optimizer
    double actu = exp(-mu(0) * T / nStep);
    boost::shared_ptr< OptimizeFictitiousSwing<BasketCall> >optimizer = boost::make_shared< OptimizeFictitiousSwing<BasketCall> >(payoff, N, p_ndim, actu);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(p_ndim, 0.);
    int initialRegime = 0;
    std::string fileToDump = "CondExp";
    bool bOneFile = false;
    boost::tuple<bool, double > valueParal = DynamicProgrammingByRegression(simulator, grid, optimizer, nbMesh, vFunction, initialStock, initialRegime, fileToDump, bOneFile);
    if (valueParal.get<0>())
    {
        BOOST_CHECK_EQUAL(valueSeq, valueParal.get<1>());
    }

    std::string fileToDumpThread = "CondExpThread";
    if (world.rank() == 0)
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // using continuation values
        double valueThread =  DynamicProgrammingByRegressionThread(simulator, grid, optimizer, nbMesh, vFunction, initialStock, initialRegime, fileToDumpThread);
        std::cout << " valueSeq " << valueSeq << " valueThread " << valueThread << std::endl ;
        BOOST_CHECK_EQUAL(valueSeq, valueThread);
    }
}

BOOST_AUTO_TEST_CASE(testSwingOption2D)
{
    testMultiStock(2);
}

// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
