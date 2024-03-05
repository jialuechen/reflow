// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <functional>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/semilagrangien/InitialValue.h"


using namespace libflow;
using namespace Eigen;
using namespace  std;


InitialValue::InitialValue(const  shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime): m_pGridCurrent(p_pGridCurrent), m_nDim(p_pGridCurrent->getDimension()), m_nbRegime(p_nbRegime)
{}

vector<shared_ptr< ArrayXd > >  InitialValue::operator()(const function<double(const int &, const ArrayXd &)>      &p_funcValue) const
{
    vector<shared_ptr< ArrayXd > > initialValues(m_nbRegime);
    if (m_pGridCurrent->getNbPoints() > 0)
    {
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
            initialValues[iReg] = make_shared<ArrayXd>(m_pGridCurrent->getNbPoints());
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
                    (*initialValues[iReg])(iterGridPoint[iThread]->getCount()) = p_funcValue(iReg, pointCoord);
                iterGridPoint[iThread]->nextInc(nbThreads);
            }
        }
    }
    return initialValues;
}

