
#ifndef LINEARINTERPOLATOR_H
#define LINEARINTERPOLATOR_H
#include <vector>
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"

namespace libflow
{

/// \class LinearInterpolator LinearInterpolator.h
/// Linear interpolation object for full grid
class LinearInterpolator : public Interpolator
{
private :

    int m_nbWeigth ; ///< storing \f$ 2^N \f$ where \f$ N \f$ is the dimension of the problem
    Eigen::Array< std::pair<double, int>, Eigen::Dynamic, 1  > m_weightAndPoints ; ///< For interpolation stores the weights and the points number in the mesh

public :

    /** \brief Default constructor
     */
    LinearInterpolator() {}

    /** \brief Constructor
     *  \param p_grid   is the grid used to interpolate
     *  \param p_point  is the coordinate of the points used for interpolatation
     */
    LinearInterpolator(const FullGrid    *p_grid, const Eigen::ArrayXd &p_point):
        m_nbWeigth(0x01 << p_point.size()), m_weightAndPoints(m_nbWeigth)
    {
        // coordinate min
        Eigen::ArrayXi  coordmin = p_grid->lowerPositionCoord(p_point);
        Eigen::ArrayXd  meshSize = p_grid->getMeshSize(coordmin);
        // get back real coordinates
        Eigen::ArrayXd xCoord = p_grid->getCoordinateFromIntCoord(coordmin);
        // weight in each direction
        Eigen::ArrayXd weightPerDim(p_point.size());
        for (int id = 0; id < p_point.size(); ++id)
        {
            // weights have to be positive : so force in case is nearly 0 (rounding error)
            // they have to be below 1
            weightPerDim(id) = std::min(std::max(0., (p_point(id) - xCoord(id)) / meshSize(id)), 1.);
        }
        Eigen::ArrayXi iCoord(p_point.size()) ;
        // iterate on all vertex of the hypercube
        for (int j = 0 ; j < m_nbWeigth ; ++j)
        {
            unsigned int ires = j ;
            double weightLocal  = 1. ;
            for (int id = p_point.size() - 1 ; id >= 0  ; --id)
            {
                unsigned int idec = (ires >> id) ;
                iCoord(id) = coordmin(id) + idec;
                weightLocal *= (1 - weightPerDim(id)) * (1 - idec) + weightPerDim(id) * idec ;
                ires -= (idec << id);
            }
            m_weightAndPoints(j) = std::make_pair(weightLocal, p_grid->intCoordPerDimToGlobal(iCoord));
        }
    }

    /**  \brief  interpolate
     *  \param  p_dataValues   Values of the data on the grid
     *  \return interpolated value
     */
    inline double apply(const Eigen::Ref< const Eigen::ArrayXd >   &p_dataValues) const
    {
        double retInterp = 0.;
        for (int i = 0; i < m_weightAndPoints.size(); ++i)
            retInterp  += m_weightAndPoints(i).first *  p_dataValues(m_weightAndPoints(i).second);
        return retInterp;
    }

    /**  \brief  interpolate and use vectorization
     *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
     *  \return interpolated value
     */
    Eigen::ArrayXd applyVec(const Eigen::ArrayXXd &p_dataValues) const
    {
        Eigen::ArrayXd retInterp(Eigen::ArrayXd::Zero(p_dataValues.rows()));

        for (int i = 0; i < m_weightAndPoints.size(); ++i)
            retInterp  += m_weightAndPoints(i).first *  p_dataValues.col(m_weightAndPoints(i).second);
        return retInterp;
    }

    /** \brief  Same as above but avoids copy for Numpy eigen mapping due to storage conventions
     *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
     *  \return interpolated value
     */
    inline Eigen::ArrayXd applyVecPy(Eigen::Ref< Eigen::ArrayXXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> >  p_dataValues) const override
    {
        Eigen::ArrayXd retInterp(Eigen::ArrayXd::Zero(p_dataValues.rows()));

        for (int i = 0; i < m_weightAndPoints.size(); ++i)
            retInterp  += m_weightAndPoints(i).first *  p_dataValues.col(m_weightAndPoints(i).second);
        return retInterp;
    }
};
}
#endif
