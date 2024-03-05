#ifdef USE_MPI

#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/dp/FinalStepDPDist.h"


using namespace libflow;
using namespace Eigen;
using namespace std;


FinalStepDPDist::FinalStepDPDist(const  shared_ptr<FullGrid> &p_pGridCurrent, const int &p_nbRegime,
                                 const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,
                                 const boost::mpi::communicator &p_world): m_pGridCurrent(p_pGridCurrent),
    m_nDim(p_pGridCurrent->getDimension()),
    m_nbRegime(p_nbRegime)
{
    // initial dimension
    ArrayXi initialDimension   = p_pGridCurrent->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_bdimToSplit, p_world);
    // grid treated by current processor
    m_gridCurrentProc = m_pGridCurrent->getSubGrid(paraSplitComputationGridsProc(initialDimension, splittingRatio, p_world.rank()));
}

vector<shared_ptr< ArrayXXd > >  FinalStepDPDist::operator()(const function<double(const int &, const ArrayXd &, const ArrayXd &)>     &p_funcValue,
        const ArrayXXd &p_particles) const
{
    vector<shared_ptr< ArrayXXd > > finalValues(m_nbRegime);
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            finalValues[iReg] = make_shared<ArrayXXd>(p_particles.cols(), m_gridCurrentProc->getNbPoints());
        shared_ptr<GridIterator>  iterGrid =  m_gridCurrentProc->getGridIterator();
        while (iterGrid->isValid())
        {
            ArrayXd pointCoord = iterGrid->getCoordinate();
            for (int iReg = 0; iReg < m_nbRegime; ++iReg)
                for (int is = 0; is < p_particles.cols(); ++is)
                {
                    (*finalValues[iReg])(is, iterGrid->getCount()) = p_funcValue(iReg, pointCoord, p_particles.col(is));

                }
            iterGrid->next();
        }
    }
    else
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            finalValues[iReg] = make_shared<ArrayXXd>();
    }
    return finalValues;
}
#endif

