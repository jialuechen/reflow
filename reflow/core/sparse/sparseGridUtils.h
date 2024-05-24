
#ifndef SPARSEGRIDUTILS_H
#define SPARSEGRIDUTILS_H
#include <array>
#include "reflow/core/utils/comparisonUtils.h"

/** \file  sparseGridUtils.h
 *  \brief utilitarian for sparse grids : permits to avoid calculations
 *  \author Xavier Warin
 */
namespace reflow
{
/// \defgroup sparseUtils Utilitary for sparse grids
/// \brief Utilitarian for sparse grids
///@{
extern std::array<double, 23> deltaSparseMesh ; ///< size or semi size of the mesh by sparse grids
extern std::array<unsigned int, 21> lastNode;  ///< node of last node associated to each level
extern std::array<int, 4> iNodeToFunc ; ///< helper for cubic hierarchization or dehierarchization

/// \brief  for left and right Hierarchization, give weight for parent for cubic Hierarchization or Dehierarchization
///@{
extern std::array<double, 2> weightParent;
extern std::array<double, 2> weightGrandParent;
extern std::array<double, 2> weightQuadraticParent;
///@}


/// \brief function used to get genericity of sparse algorithms
///        Either  the input is an Eigen::ArrayXd and the return is a double
///        or it is an Eigen::ArrayXXd and the return is an Eigen::ArrayXd
class DoubleOrArray
{
public:

    /// \brief First operator()
    /// \param  p_x       array used to get values
    /// \param  p_point   index in the array
    /// \return  p_x(p_point)
    inline double operator()(const Eigen::ArrayXd &p_x, const int &p_point) const
    {
        return p_x(p_point);
    }
    /// \brief Second operator ()
    /// \param p_x  array used to get values
    /// \param  p_point  index in the array
    /// \return  p_x(p_point)
    inline Eigen::ArrayXd operator()(const Eigen::ArrayXXd &p_x, const int &p_point) const
    {
        return p_x.col(p_point) ;
    }

    /// \brief Zero affectation
    /// \param p_x vector to assign
    ///
    inline void zero(double &p_x, const Eigen::ArrayXd &) const
    {
        p_x = 0;
    }

    /// \brief Zero affectation for array
    /// \param p_x vector to assign
    /// \param  p_ut  utilitarian to get size of object
    inline void zero(Eigen::ArrayXd &p_x, const Eigen::ArrayXXd &p_ut) const
    {
        p_x = Eigen::ArrayXd::Zero(p_ut.rows());
    }


    /// \brief affectation operator
    /// \param p_x  array used to get values
    /// \param p_point  index in the array
    /// \param p_affect value to affect
    inline void affect(Eigen::ArrayXd &p_x, const int &p_point, const double &p_affect)  const
    {
        p_x(p_point) = p_affect;
    }

    /// \brief second affectation operator
    /// \param p_x  array used to get values
    /// \param p_point  index in the array
    /// \param p_affect value to affect
    inline void affect(Eigen::ArrayXXd &p_x, const int &p_point, const Eigen::ArrayXd   &p_affect)  const
    {
        p_x.col(p_point) = p_affect;
    }

    /// \brief resize operator
    /// \param p_x  array to work on
    /// \param p_size size to impose to the array
    inline void resize(Eigen::ArrayXd &p_x, int p_size)
    {
        p_x.conservativeResize(p_size);
    }

    /// \brief resize operator
    /// \param p_x  array to work on
    /// \param p_size size of the array
    inline void resize(Eigen::ArrayXXd &p_x, int p_size)
    {
        int nRows = p_x.rows();
        p_x.conservativeResize(nRows, p_size);
    }
};


/// \brief Evaluate linear function basis
///        Basis on \f$ [ xMilde -1./unDx, xMilde+1./unDx]\f$
///        Hat function taking value 1 in xMilde, 0 in \f$ xMilde -1./unDx, xMilde+1./unDx \f$
///        Evaluation for \f$  x \in [ xMilde -1./unDx, xMilde+1./unDx] \f$
class LinearHatValue
{

    double m_xMilde; ///<  coordinate of the middle of the mesh
    double m_unDx ; ///< one over size of mesh
    double m_scale ; ///< scaling factor
public :

    /// \brief Constructor
    /// \param p_xMidle coordinate of the middle of the mesh
    /// \param p_unDx   one over size of mesh
    LinearHatValue(const double &p_xMidle, const double &p_unDx) : m_xMilde(p_xMidle), m_unDx(p_unDx), m_scale(1.) {}

    /// \brief Constructor
    /// \param p_xMidle coordinate of the middle of the mesh
    /// \param p_unDx   one over size of mesh
    /// \param p_scale   scaling factor
    LinearHatValue(const double &p_xMidle, const double &p_unDx, const double &p_scale) : m_xMilde(p_xMidle), m_unDx(p_unDx), m_scale(p_scale) {}

    /// \brief operator ()
    /// \param p_x  Point coordinate where the function is evaluated
    inline double operator()(const double &p_x) const
    {
        assert(isLesserOrEqual(std::fabs(m_xMilde - p_x)*m_unDx, 1.));
        return m_scale * (1. - std::fabs(m_xMilde - p_x) * m_unDx);
    }
};


/// \brief Evaluate quadratic function basis
///        Basis on \f$ [ xMilde -1./unDx, xMilde+1./unDx]\f$
///        Evaluation for \f$  x \in [ xMilde -1./unDx, xMilde+1./unDx] \f$
class QuadraticValue
{
    double m_xMilde; ///<  coordinate of the middle of the mesh
    double m_unDx ; ///< one over size of mesh
public :

    /// \brief Constructor
    /// \param p_xMidle coordinate of the middle of the mesh
    /// \param p_unDx   one over size of mesh
    QuadraticValue(const double &p_xMidle, const double &p_unDx): m_xMilde(p_xMidle), m_unDx(p_unDx) {}

    /// \brief operator ()
    /// \param p_x  Point coordinate where the function is evaluated
    inline double operator()(const double   &p_x) const
    {
        assert(isLesserOrEqual(std::fabs(m_xMilde - p_x)*m_unDx, 1.));
        double xInter = (m_xMilde - p_x) * m_unDx;
        return (1 + xInter) * (1 - xInter);
    }
};


/// \brief Evaluate Cubic function basis (left one)
class CubicLeftValue
{
private :

    double m_unTiers;
    double m_xMilde; ///<  coordinate of the middle of the mesh
    double m_unDx ; ///< one over size of mesh

public :

    /// \brief Constructor
    /// \param p_xMidle coordinate of the middle of the mesh
    /// \param p_unDx   one over size of mesh
    CubicLeftValue(const double &p_xMidle, const double &p_unDx): m_unTiers(1. / 3.), m_xMilde(p_xMidle), m_unDx(p_unDx) {}

    /// \brief operator ()
    /// \param p_x  Point coordinate where the function is evaluated
    inline double operator()(const double &p_x) const
    {
        double xInter = (p_x - m_xMilde) * m_unDx;
        if (std::fabs(xInter) > 1)
            return 0.;
        return (xInter * xInter - 1) * (xInter - 3) * m_unTiers;
    }
};


/// \brief Evaluate Cubic function basis (right one)
class CubicRightValue
{
private :

    double unTiers;
    double m_xMilde; ///<  coordinate of the middle of the mesh
    double m_unDx ; ///< one over size of mesh

public :

    /// \brief Constructor
    /// \param p_xMidle coordinate of the middle of the mesh
    /// \param p_unDx   one over size of mesh
    CubicRightValue(const double &p_xMidle, const double &p_unDx): unTiers(1. / 3.), m_xMilde(p_xMidle), m_unDx(p_unDx) {}

    /// \brief operator ()
    /// \param p_x  Point coordinate where the function is evaluated
    inline double operator()(const double &p_x) const
    {
        double xInter = (p_x - m_xMilde) * m_unDx;
        return (1. - xInter * xInter) * (xInter + 3) * unTiers;
    }
};

/// \brief One function
class OneFunction
{

public :

    OneFunction() {}

    /// \brief return one
    inline double operator()(const double &) const
    {
        return 1;
    }
};


/// \brief Permits to get non direct father of a point for a sparse grid without boundary points
class GetNonDirectFatherNoBound
{
public :
    /// \brief Constructor
    GetNonDirectFatherNoBound() {}

    /// \param p_level             level  (to update)
    /// \param p_position          position (to update)
    void inline operator()(char   &p_level,   unsigned int   &p_position)
    {
        assert(p_level > 2);
        unsigned int  ip = p_position & 1; // 1 if even, 0 otherwise
        unsigned int  father = p_position >> 1;
        p_position = (father >> 1); // grandfather
        p_level -= 2 ;
        // test if son is correct
        while (((p_position << 1) | ip) == father)
        {
            father = p_position;
            p_position = p_position >> 1;
            p_level -= 1 ;
        }
    }
};

/// \brief Permits to get non direct father of a point for a sparse grid with boundary points
class GetNonDirectFatherBound
{
public :
    /// \brief Constructor
    GetNonDirectFatherBound() {}

    /// \param p_level             level  (to update)
    /// \param p_position          position (to update)
    void inline operator()(char   &p_level,   unsigned int   &p_position)
    {
        assert(p_level > 1);
        if (p_position == 0)
        {
            p_level = 1 ;
            p_position = 0;
            return ;
        }
        else if (p_position == lastNode[p_level - 1])
        {
            p_level = 1 ;
            p_position = 2;
            return ;
        }
        unsigned int  ip = p_position & 1; // 1 if even, 0 otherwise
        unsigned int  father = p_position >> 1;
        p_position = (father >> 1); // grandfather
        p_level -= 2 ;
        // test if son is correct
        while (((p_position << 1) | ip) == father)
        {
            father = p_position;
            p_position = p_position >> 1;
            p_level -= 1 ;
        }
        // correct if first level
        if (p_level == 1)
            p_position = 1 ;
    }
};

/// \brief permits to get direct father for sparse grids without boundary points
class GetDirectFatherNoBound
{
public :
    /// \brief Constructor
    GetDirectFatherNoBound() {}

    /// \param p_level                level  (to update)
    /// \param p_position              position (to update)
    void inline operator()(char   &p_level,   unsigned int   &p_position)
    {
        p_level -= 1 ;
        p_position = (p_position >> 1) ;
    }
};

/// \brief permits to get direct father for sparse grids with boundary points
class GetDirectFatherBound
{
public :
    /// \brief Constructor
    GetDirectFatherBound() {}

    /// \param p_level                level  (to update)
    /// \param p_position              position (to update)
    void inline operator()(char   &p_level,   unsigned int   &p_position)
    {
        assert(p_level > 1);
        if (p_level == 2)
        {
            p_level = 1 ;
            p_position = 1 ; // centrer point
        }
        else
        {
            p_level -= 1 ;
            p_position = (p_position >> 1) ;
        }
    }
};
///@}
}

#endif /* SPARSEGRIDUTILS.H */
