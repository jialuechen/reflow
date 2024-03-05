
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/core/grids/LegendreInterpolatorSpectral.h"

using namespace libflow;
using namespace Eigen;
using namespace std;

LegendreInterpolatorSpectral::LegendreInterpolatorSpectral(const RegularLegendreGrid *p_grid, const ArrayXd &p_values) :  m_grid(p_grid),
    m_min(ArrayXd::Constant(p_grid->getNbMeshes(), infty)),
    m_max(ArrayXd::Constant(p_grid->getNbMeshes(), -infty))
{

    shared_ptr< vector< ArrayXXd >  >  fInterpol = p_grid->getFInterpol();
    const ArrayXi &nbStep = p_grid->getNbStep();
    ArrayXi poly = p_grid->getPoly();
    m_funcBaseExp.resize(poly.size(), (1 + poly).prod());
    // utilization
    ArrayXi  imult(poly.size()) ;
    imult(0) = 1;
    for (int id = 1; id < poly.size(); ++id)
        imult(id) =  imult(id - 1) * (poly(id - 1) + 1);
    for (int i = 0 ; i < m_funcBaseExp.cols() ; ++i)
    {
        int irest = i ;
        for (int id = poly.size() - 1 ; id >= 0; --id)
        {
            m_funcBaseExp(id, i) = irest / imult(id);
            irest = irest % imult(id);
        }
    }
    // resize and calculate spectral representation
    m_spectral.resize(m_funcBaseExp.cols(), p_grid->getNbMeshes());
    m_spectral.setConstant(0.);
    // utilitarian for coordinates
    ArrayXi coordMesh(poly.size());
    ArrayXi coordPoint(poly.size());
    for (int imesh = 0 ; imesh < p_grid->getNbMeshes(); ++imesh)
    {
        int imeshLoc = imesh;
        for (int id = 0; id < poly.size(); ++id)
        {
            int idec = nbStep(id);
            coordMesh(id) = imeshLoc % idec;
            imeshLoc /= idec;
        }
        // translate to point coordinates
        coordMesh *= poly;
        // nest on collocation  point
        for (int ibb = 0 ; ibb < m_funcBaseExp.cols(); ++ibb)
        {
            coordPoint = coordMesh + m_funcBaseExp.col(ibb);
            // global coordinate
            int ipoint = p_grid->intCoordPerDimToGlobal(coordPoint);
            for (int ib = 0 ; ib < m_funcBaseExp.cols(); ++ib)
            {
                double dfunc = 1 ;
                for (int id = 0 ; id < poly.size() ; ++id)
                    dfunc *= (*fInterpol)[id](m_funcBaseExp(id, ib), m_funcBaseExp(id, ibb));
                m_spectral(ib, imesh) += p_values(ipoint) * dfunc;
            }
            m_min(imesh) = min(m_min(imesh), p_values(ipoint));
            m_max(imesh) = max(m_max(imesh), p_values(ipoint));
        }

    }
}

