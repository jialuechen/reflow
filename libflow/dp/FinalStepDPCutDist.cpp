#ifdef USE_MPI

#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/dp/FinalStepDPCutDist.h"


using namespace libflow;
using namespace Eigen;
using namespace std;


FinalStepDPCutDist::FinalStepDPCutDist(const  shared_ptr<FullGrid> &p_pGridCurrent, const int &p_nbRegime,
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

vector<shared_ptr< ArrayXXd > >  FinalStepDPCutDist::operator()(const function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>     &p_funcValue,
        const ArrayXXd &p_particles) const
{
    vector<shared_ptr< ArrayXXd > > finalValues(m_nbRegime);
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        int ndimCut = m_gridCurrentProc->getDimension() + 1;
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            finalValues[iReg] = make_shared<ArrayXXd>(p_particles.cols() * ndimCut, m_gridCurrentProc->getNbPoints());
        shared_ptr<GridIterator>  iterGrid =  m_gridCurrentProc->getGridIterator();
        while (iterGrid->isValid())
        {
            ArrayXd pointCoord = iterGrid->getCoordinate();
            int ipoint = iterGrid->getCount();
            for (int iReg = 0; iReg < m_nbRegime; ++iReg)
                for (int is = 0; is < p_particles.cols(); ++is)
                {
                    ArrayXd cuts = p_funcValue(iReg, pointCoord, p_particles.col(is));
                    for (int ic = 0; ic < ndimCut; ++ic)
                        (*finalValues[iReg])(is + ic * p_particles.cols(), ipoint) = cuts(ic);

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

