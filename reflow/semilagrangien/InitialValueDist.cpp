#ifdef USE_MPI
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/core/utils/primeNumber.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/grids/GridIterator.h"
#include "reflow/semilagrangien/InitialValueDist.h"


using namespace reflow;
using namespace Eigen;
using namespace std;


InitialValueDist::InitialValueDist(const  shared_ptr<FullGrid> &p_pGridCurrent, const int &p_nbRegime,
                                   const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,
                                   const boost::mpi::communicator &p_world): m_pGridCurrent(p_pGridCurrent), m_nDim(p_pGridCurrent->getDimension()), m_nbRegime(p_nbRegime)
{
    // initial dimension
    ArrayXi initialDimension   = p_pGridCurrent->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_bdimToSplit, p_world);
    // grid treated by current processor
    m_gridCurrentProc = m_pGridCurrent->getSubGrid(paraSplitComputationGridsProc(initialDimension, splittingRatio, p_world.rank()));
}

vector<shared_ptr< ArrayXd > >  InitialValueDist::operator()(const function<double(const int &, const ArrayXd &)>   &p_funcValue) const
{
    vector<shared_ptr< ArrayXd > > initialValues(m_nbRegime);
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            initialValues[iReg] = make_shared<ArrayXd>(m_gridCurrentProc->getNbPoints());
        shared_ptr<GridIterator>  iterGrid =  m_gridCurrentProc->getGridIterator();
        while (iterGrid->isValid())
        {
            ArrayXd pointCoord = iterGrid->getCoordinate();
            for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            {
                (*initialValues[iReg])(iterGrid->getCount()) = p_funcValue(iReg, pointCoord);

            }
            iterGrid->next();
        }
    }
    return initialValues;
}
#endif

