// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include "libflow/core/parallelism/ParallelHiter.h"

using namespace libflow ;
using namespace Eigen;


ParallelHiter::ParallelHiter(const Ref<const Array< std::array<int, 2 >, Dynamic, 1 > >    &p_hBounds,
                             const Ref<const Array< std::array<int, 2 >, Dynamic, 1 > >    &p_hBoundsGlob):
    m_bValid(true), m_hBounds(p_hBounds), m_hBoundsGlob(p_hBoundsGlob),
    m_hVal(p_hBounds.size()), m_strides(p_hBounds.size())
{
    m_hVal.setConstant(p_hBounds.size());
    for (int i = 0; i < p_hBounds.size(); i++)
    {
        m_hVal(i) = m_hBounds(i)[0] ;
    }
    m_strides(0) = 1;
    for (int dim = 1; dim < m_hBounds.size() ; ++dim)
    {
        m_strides[dim] = m_strides[dim - 1 ] * (m_hBoundsGlob(dim - 1)[1] - m_hBoundsGlob(dim - 1)[0]);
    }
}


size_t ParallelHiter::hCubeSize()
{
    int size = 1;

    for (int d = 0; d < m_hBounds.size(); d++)
    {
        size *= (m_hBounds(d)[1] - m_hBounds(d)[0]);
    }
    return (size);
}


bool ParallelHiter::next(void)
{
    int idx = 1;
    bool done = false;
    while ((idx < m_hBounds.size()) && !done)
    {
        if (m_hVal(idx) < (m_hBounds(idx)[1] - 1))
        {
            m_hVal(idx)++;
            done = true;
        }
        else
        {
            m_hVal(idx) = m_hBounds(idx)[0];
            idx++;
        }
    }
    m_bValid = done;
    // Return the result of the reset
    return (done);
}

