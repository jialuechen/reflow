
#ifndef LEGENDREINTERPOLATOR_H
#define LEGENDREINTERPOLATOR_H
#include <vector>
#include <Eigen/Dense>
#include "reflow/core/grids/Interpolator.h"


namespace reflow
{
class RegularLegendreGrid;

class LegendreInterpolator : public Interpolator
{
private :

    int m_nbWeigth ; ///< storing the number of points involved in representation
    Eigen::Array< std::pair<double, int>, Eigen::Dynamic, 1  > m_weightAndPoints ; ///< For interpolation stores the weights and the points number in the mesh

public :

    /** \brief Default constructor
     */
    LegendreInterpolator() {}

    /** \brief Constructor taking the coordinates
     *  \param p_grid   is the grid used to interpolate
     *  \param p_point  is the coordinate of the points used for interpolation
     */
    LegendreInterpolator(const   RegularLegendreGrid *p_grid, const Eigen::ArrayXd &p_point);

    /**  \brief  interpolate
     *  \param  p_dataValues   Values of the data on the grid
     *  \return interpolated value
     */
    double apply(const Eigen::Ref< const Eigen::ArrayXd > &p_dataValues) const
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
