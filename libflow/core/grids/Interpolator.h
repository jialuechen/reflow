
#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H
#include <vector>
#include <Eigen/Dense>

namespace libflow
{

/// \class Interpolator Interpolator.h
///  Interpolation  base class
class Interpolator
{
public :

    /// \brief Default constructor
    Interpolator() {}

    /// \brief Default Destructor
    virtual ~Interpolator() {}

    /**  \brief  interpolate
     *  \param  p_dataValues   Values of the data on the grid
     *  \return interpolated value
     */
    virtual  double apply(const Eigen::Ref< const Eigen::ArrayXd > &p_dataValues) const = 0;

    /**  \brief  interpolate and use vectorization
    *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
    *  \return interpolated value
    */
    virtual  Eigen::ArrayXd applyVec(const Eigen::ArrayXXd &p_dataValues) const = 0;

    /** \brief  Same as above but avoids copy for Numpy eigen mapping due to storage conventions
     *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
     *  \return interpolated value
     */
    virtual  Eigen::ArrayXd applyVecPy(Eigen::Ref< Eigen::ArrayXXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> >  p_dataValues) const = 0;

};
}
#endif
