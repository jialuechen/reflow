// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <vector>
#include <memory>
#include <functional>
#include <array>
#include <math.h>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/grids/RegularGrid.h"
#include "libflow/core/grids/FullLegendreGridIterator.h"
#include "libflow/core/grids/LegendreInterpolator.h"
#include "libflow/core/grids/LegendreInterpolatorSpectral.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "libflow/core/utils/AnalyticLegendre.h"


using namespace libflow;
using namespace Eigen;
using namespace std;


RegularLegendreGrid::RegularLegendreGrid(const ArrayXd &p_lowValues, const ArrayXd &p_step, const  ArrayXi &p_nbStep, const ArrayXi &p_poly):
    RegularGrid(p_lowValues, p_step, p_nbStep), m_poly(p_poly),
    m_gllPoints(p_lowValues.size()),
    m_legendre(new array< function< double(const double &) >, 11 >
{
    {
        Legendre0(), Legendre1(), Legendre2(), Legendre3(), Legendre4(), Legendre5(), Legendre6(), Legendre7(), Legendre8(), Legendre9(), Legendre10()
    }
}),
m_fInterpol(make_shared< vector< ArrayXXd > >(p_poly.size())), m_firstPoints(ArrayXi::Zero(p_lowValues.size())), m_lastPoints(p_lowValues.size())
{
    m_lastPoints =  p_poly ;
    for (size_t id = 0; id < m_gllPoints.size(); ++id)
    {
        m_gllPoints[id].resize(p_poly(id) + 1);
        m_gllPoints[id](0) = -1. ;
        m_gllPoints[id](p_poly(id)) = 1. ;
        if (p_poly(id) == 2)
            m_gllPoints[id](1) = 0. ;
        else if (p_poly(id) > 2)
        {
            int n = p_poly(id) - 1;
            MatrixXd mat = MatrixXd::Zero(n, n);
            for (int i = 0; i < n - 1; ++i)
            {
                double util = 0.5 * sqrt((i + 1) * (i + 3.) / ((i + 1.5) * (i + 2.5)));
                mat(i + 1, i) = util;
                mat(i, i + 1) = util;
            }
            SelfAdjointEigenSolver<MatrixXd> es(mat);
            // eigen values
            m_gllPoints[id].segment(1, p_poly(id) - 1) = es.eigenvalues().array();
        }
    }
    // function used for interpolations
    for (int id = 0; id < p_poly.size(); ++id)
    {
        (*m_fInterpol)[id].resize(p_poly(id) + 1, p_poly(id) + 1);
        ArrayXd rho(p_poly(id) + 1);
        for (int i = 0; i <= p_poly(id); ++i)
            rho(i) = 2. / ((p_poly(id) + 1) * p_poly(id) * pow((*m_legendre)[p_poly(id)](m_gllPoints[id](i)), 2.));
        for (int k = 0; k <= p_poly(id); ++k)
        {
            double weight = 0;
            for (int i = 0; i <= p_poly(id); ++i)
                weight += pow((*m_legendre)[k](m_gllPoints[id](i)), 2.) * rho(i);
            weight = 1. / weight;
            for (int i = 0; i <= p_poly(id); ++i)
            {

                (*m_fInterpol)[id](k, i) =  weight * rho(i) * (*m_legendre)[k](m_gllPoints[id](i));
            }
        }
    }
    // update dimension
    m_dimensions = m_nbStep * p_poly + 1;
    m_nbPoints = m_dimensions.prod();
}

RegularLegendreGrid::RegularLegendreGrid(const ArrayXd &p_lowValues, const ArrayXd &p_step, const  ArrayXi &p_nbStep,  const   vector< ArrayXd >   &p_gllPoints,
        shared_ptr< vector< ArrayXXd >  > p_fInterpol, const ArrayXi &p_firstPoints,  const ArrayXi &p_lastPoints):
    RegularGrid(p_lowValues, p_step, p_nbStep), m_poly(p_lowValues.size()), m_gllPoints(p_gllPoints),
    m_legendre(new array< function< double(const double &) >, 11 >
{
    {
        Legendre0(), Legendre1(), Legendre2(), Legendre3(), Legendre4(), Legendre5(), Legendre6(), Legendre7(), Legendre8(), Legendre9(), Legendre10()
    }
}),
m_fInterpol(p_fInterpol), m_firstPoints(p_firstPoints), m_lastPoints(p_lastPoints)
{
    for (size_t id = 0; id < m_gllPoints.size(); ++id)
    {
        m_poly(id) = p_gllPoints[id].size() - 1;
    }
    // update dimension
    m_dimensions = m_nbStep * m_poly + 1 - m_firstPoints - (m_poly - m_lastPoints);
    m_nbPoints = m_dimensions.prod();
}



ArrayXi RegularLegendreGrid::lowerPositionCoord(const Ref<const ArrayXd > &p_point) const
{
#ifndef NOCHECK_GRID
    assert(isInside(p_point)) ;
#endif
    ArrayXi intCoord(p_point.size());
    for (int i = 0; i < p_point.size(); ++i)
    {
        intCoord(i) = max(min(roundIntAbove((p_point(i) - m_lowValues(i)) / m_step(i)), m_nbStep(i) - 1), 0) * m_poly(i);
    }
    return intCoord;
}

ArrayXi RegularLegendreGrid::upperPositionCoord(const Ref<const ArrayXd > &p_point) const
{
#ifndef NOCHECK_GRID
    assert(isInside(p_point)) ;
#endif
    ArrayXi intCoord(p_point.size());
    for (int i = 0; i < p_point.size(); ++i)
    {
        intCoord(i) = max(min(roundIntAbove((p_point(i) - m_lowValues(i)) / m_step(i)) + 1, m_nbStep(i)) * m_poly(i), 0);
    }
    return intCoord;
}


ArrayXd  RegularLegendreGrid::getCoordinateFromIntCoord(const ArrayXi &p_icoord) const
{
    ArrayXd ret(p_icoord.size());
    for (int i = 0; i < p_icoord.size(); ++i)
    {
        int nPoint = m_poly(i);
        int coordRec =  p_icoord(i) + m_firstPoints(i);
        int imesh = coordRec / nPoint;
        int ipoint = coordRec % nPoint;
        ret(i) =  m_lowValues(i) +  m_step(i) * (imesh + 0.5 * (1 + m_gllPoints[i](ipoint)));
    }
    return ret;
}

shared_ptr<FullGrid> RegularLegendreGrid::getSubGrid(const Array< array<int, 2>, Dynamic, 1> &p_mesh) const
{
    if (p_mesh.size() == 0)
    {
        return   make_shared<RegularLegendreGrid>();
    }
    ArrayXd lowValues(p_mesh.size()) ;
    ArrayXi  nbStep(p_mesh.size()) ;
    ArrayXi firstPoint(p_mesh.size());
    ArrayXi lastPoint(p_mesh.size());
    for (int id = 0; id < p_mesh.size(); ++id)
    {
        firstPoint(id) = p_mesh(id)[0] % m_poly(id);
        lastPoint(id) = (p_mesh(id)[1] - 1) % m_poly(id);
        if (lastPoint(id) == 0)
            lastPoint(id) = m_poly(id);
        nbStep(id) = (p_mesh(id)[1] - 1 + m_poly(id) - lastPoint(id) - p_mesh(id)[0] + firstPoint(id)) / m_poly(id);
        lowValues(id) = m_lowValues(id) + p_mesh(id)[0] / m_poly(id) * m_step(id);
    }
    return make_shared<RegularLegendreGrid>(lowValues, m_step, nbStep, m_gllPoints, m_fInterpol, firstPoint, lastPoint);
}


shared_ptr< GridIterator> RegularLegendreGrid::getGridIteratorInc(const int   &p_iThread) const
{
    return make_shared<FullLegendreGridIterator>(m_lowValues, m_step, m_nbStep, m_poly, m_gllPoints, m_firstPoints, m_lastPoints, p_iThread) ;
}


shared_ptr< GridIterator> RegularLegendreGrid::getGridIterator() const
{
    return make_shared<FullLegendreGridIterator>(m_lowValues, m_step,  m_nbStep, m_poly, m_gllPoints, m_firstPoints, m_lastPoints) ;
}

shared_ptr<Interpolator> RegularLegendreGrid::createInterpolator(const ArrayXd &p_coord) const
{
    return  make_shared<LegendreInterpolator>(this, p_coord) ;

}

shared_ptr<InterpolatorSpectral> RegularLegendreGrid::createInterpolatorSpectral(const Eigen::ArrayXd &p_values) const
{
    return  make_shared<LegendreInterpolatorSpectral>(this, p_values);
}

void RegularLegendreGrid::rescalepoint(const double  &p_point, const int &p_idim,  double &p_coordLoc, int &p_iCoord)const
{
    p_iCoord = max(min(roundIntAbove((p_point - m_lowValues(p_idim)) / m_step(p_idim)), m_nbStep(p_idim) - 1), 0) * m_poly(p_idim);
    int coordRec =  p_iCoord + m_firstPoints(p_idim);
    int imesh = coordRec / m_poly(p_idim);
    int ipoint = coordRec % m_poly(p_idim);
    double coordMin =  m_lowValues(p_idim) +  m_step(p_idim) * (imesh + 0.5 * (1 + m_gllPoints[p_idim](ipoint)));
    p_coordLoc = 2.*(p_point - coordMin) /  m_step(p_idim) - 1.;
}
