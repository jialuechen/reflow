
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/FinalStepDP.h"


using namespace libflow;
using namespace Eigen;
using namespace std;

FinalStepDP::FinalStepDP(const  shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime): m_pGridCurrent(p_pGridCurrent), m_nDim(p_pGridCurrent->getDimension()), m_nbRegime(p_nbRegime)
{}


vector<shared_ptr< ArrayXXd > >  FinalStepDP::operator()(const function<double(const int &, const ArrayXd &, const ArrayXd &)>     &p_funcValue,
        const ArrayXXd &p_particles) const
{
    vector<shared_ptr< ArrayXXd > > finalValues(m_nbRegime);
    if (m_pGridCurrent->getNbPoints() > 0)
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            finalValues[iReg] = make_shared<ArrayXXd>(p_particles.cols(), m_pGridCurrent->getNbPoints());
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
                for (int iReg = 0; iReg < m_nbRegime; ++iReg)
                    for (int is = 0; is < p_particles.cols(); ++is)
                        (*finalValues[iReg])(is, iterGridPoint[iThread]->getCount()) = p_funcValue(iReg, pointCoord, p_particles.col(is));
                iterGridPoint[iThread]->nextInc(nbThreads);
            }
        }
    }
    return finalValues;
}

