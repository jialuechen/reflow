
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "reflow/core/grids/GridIterator.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/FinalStepDPCut.h"


using namespace reflow;
using namespace Eigen;
using namespace std;

FinalStepDPCut::FinalStepDPCut(const  shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime): m_pGridCurrent(p_pGridCurrent), m_nDim(p_pGridCurrent->getDimension()), m_nbRegime(p_nbRegime)
{}


vector<shared_ptr< ArrayXXd > >  FinalStepDPCut::operator()(const function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>     &p_funcValue,
        const ArrayXXd &p_particles) const
{
    vector<shared_ptr< ArrayXXd > > finalValues(m_nbRegime);
    int ndimCut = m_pGridCurrent->getDimension() + 1;
    if (m_pGridCurrent->getNbPoints() > 0)
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            finalValues[iReg] = make_shared<ArrayXXd>(p_particles.cols() * ndimCut, m_pGridCurrent->getNbPoints());
        // number of thread
#ifdef _OPENMP
        int nbThreads = omp_get_max_threads();
#else
        int nbThreads = 1;
#endif
        // create iterator on current grid treated for processor
        vector< shared_ptr< GridIterator  > > iterGridPoint(nbThreads);
        for (int iThread = 0; iThread < nbThreads; ++iThread)
            iterGridPoint[iThread] = m_pGridCurrent->getGridIteratorInc(iThread);
        // iterates on points of the grid
        int iIter;
#ifdef _OPENMP
        #pragma omp parallel for  private(iIter)
#endif
        for (iIter = 0; iIter < static_cast<int>(m_pGridCurrent->getNbPoints()); ++iIter)
        {
#ifdef _OPENMP
            int iThread = omp_get_thread_num() ;
#else
            int iThread = 0;
#endif
            if (iterGridPoint[iThread]->isValid())
            {
                ArrayXd pointCoord = iterGridPoint[iThread]->getCoordinate();
                int ipoint = iterGridPoint[iThread]->getCount();
                for (int iReg = 0; iReg < m_nbRegime; ++iReg)
                    for (int is = 0; is < p_particles.cols(); ++is)
                    {
                        ArrayXd cuts = p_funcValue(iReg, pointCoord, p_particles.col(is));
                        for (int ic = 0; ic < cuts.size(); ++ic)
                            (*finalValues[iReg])(is + ic * p_particles.cols(), ipoint) = cuts(ic);
                    }
                iterGridPoint[iThread]->nextInc(nbThreads);
            }
        }
    }
    return finalValues;
}

