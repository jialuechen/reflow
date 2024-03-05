
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/LegendreInterpolator.h"
#include "libflow/core/grids/RegularLegendreGrid.h"


using namespace libflow ;
using namespace std;

LegendreInterpolator::LegendreInterpolator(const   RegularLegendreGrid *p_grid, const Eigen::ArrayXd &p_point)
{

    shared_ptr< array< function< double(const double &) >, 11 > > legendre = p_grid->getLegendre();
    shared_ptr< vector< Eigen::ArrayXXd >  >  fInterpol = p_grid->getFInterpol();
    Eigen::ArrayXi poly = p_grid->getPoly();
    // number of points involved
    m_nbWeigth = (1 + poly).prod();
    m_weightAndPoints.resize(m_nbWeigth);
    // coordinate min
    Eigen::ArrayXi  coordmin = p_grid->lowerPositionCoord(p_point);
    Eigen::ArrayXd  meshSize = p_grid->getMeshSize(coordmin);
    // get back real coordinates renormalized in [-1,1]
    Eigen::ArrayXd xCoord(p_point.size());
    Eigen::ArrayXd xCoordMin = p_grid->getCoordinateFromIntCoord(coordmin);
    for (int ip = 0; ip < xCoord.size(); ++ip)
        xCoord(ip) = std::max(std::min(2.*(p_point(ip) - xCoordMin(ip)) / meshSize(ip) - 1., 1.), -1.);
    // calculate weights
    Eigen::ArrayXi iCoord(p_point.size()) ; // coordinates in the mesh
    // iterate on all  points on the mesh
    for (int j = 0 ; j < m_nbWeigth ; ++j)
    {
        int jloc = j;
        int nPoint = m_nbWeigth;
        for (int id = p_point.size() - 1 ; id >= 0  ; --id)
        {
            nPoint /= (poly(id) + 1);
            iCoord(id) = jloc / nPoint;
            jloc = jloc % nPoint;
        }
        // now iterates
        //  calculate \f$\sum_i ....\sum_j L_i(\xi_{icoord(0)}) L_i(xCoord) \rho_i \kappa_i ...L_j(\xi_{icoord(nd-1)}) L_j(xCoord(nd-1)) \rho_j (j+0.5) \f$
        double weight = 0.;
        for (int jj = 0 ; jj < m_nbWeigth ; ++jj)
        {
            int jjloc = jj;
            int nnPoint = m_nbWeigth;
            double weightLocal = 1.;
            for (int id = p_point.size() - 1 ; id >= 0  ; --id)
            {
                nnPoint /= (poly(id) + 1);
                int iiCoord = jjloc / nnPoint;
                weightLocal *= (*fInterpol)[id](iiCoord, iCoord(id)) * (*legendre)[iiCoord](xCoord[id]);
                jjloc = jjloc % nnPoint;
            }
            weight += weightLocal;
        }
        m_weightAndPoints(j) = make_pair(weight, p_grid->intCoordPerDimToGlobal(iCoord + coordmin));
    }
}
