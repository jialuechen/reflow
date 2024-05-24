
#ifndef LOCALLINEARREGRESSIONFORSDDP_H
#define LOCALLINEARREGRESSIONFORSDDP_H
#include "reflow/regression/LocalLinearRegression.h"

/** \file   LocalLinearRegressionForSDDP.h
 *  \brief  Conditional expectation by local regression extended for SDDP
  *  \author Xavier Warin
 */
namespace reflow
{

/// \class LocalLinearRegressionForSDDP LocalLinearRegressionForSDDP.h
/// Special case of linear regression for SDDP
class LocalLinearRegressionForSDDP: public LocalLinearRegression
{

public :

    /// \brief Default constructor
    LocalLinearRegressionForSDDP() {}


    /// \brief First constructor for object constructed at each time step
    /// \param  p_bZeroDate    first date is 0?
    /// \param  p_particles    particles used for the meshes.
    ///                        First dimension  : dimension of the problem,
    ///                        second dimension : the  number of particles
    /// \param  p_nbMesh       discretization in each direction
    /// \param  p_bRotationAndRecale do we use SVD
    LocalLinearRegressionForSDDP(const bool &p_bZeroDate,
                                 const Eigen::ArrayXXd &p_particles,
                                 const Eigen::ArrayXi   &p_nbMesh, bool  p_bRotationAndRecale = false) : LocalLinearRegression(p_bZeroDate, p_particles, p_nbMesh, p_bRotationAndRecale)
    {
        evaluateSimulBelongingToCell();
    }

    /// \brief Second constructor
    /// \param  p_bZeroDate        first date is 0?
    /// \param  p_particles        particles used for the meshes : it is normalized et rescaled
    ///                            First dimension  : dimension of the problem,
    ///                            second dimension : the  number of particles
    /// \param  p_nbMesh           number of mesh per dimension
    /// \param  p_mesh             for each cell and each direction defines the min and max coordinate of a cell
    /// \param  p_mesh1D           meshes in each direction
    /// \param  p_simToCell        For each simulation gives its global mesh number
    /// \param  p_matReg           regression matrix factorized
    /// \param  p_diagReg          diagonal of factorized matrix
    /// \param  p_simulBelongingToCell  To each cell associate the set of all particles belonging to this cell
    /// \param  p_meanX            scaled factor in each direction (average of particles values in each direction)
    /// \param  p_etypX            scaled factor in each direction (standard deviation of particles in each direction)
    /// \param  p_svdMatrix        svd matrix transposed  used to transform particles
    /// \param  p_bRotationAndRecale do we use SVD
    LocalLinearRegressionForSDDP(const bool &p_bZeroDate,
                                 const Eigen::ArrayXXd  &p_particles,
                                 const Eigen::ArrayXi &p_nbMesh,
                                 const Eigen::Array< std::array< double, 2>, Eigen::Dynamic, Eigen::Dynamic > &p_mesh,
                                 const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_mesh1D,
                                 const Eigen::ArrayXi &p_simToCell,
                                 const Eigen::ArrayXXd   &p_matReg,
                                 const Eigen::ArrayXXd   &p_diagReg,
                                 const std::vector< std::shared_ptr< std::vector< int> > > &p_simulBelongingToCell,
                                 const Eigen::ArrayXd &p_meanX,
                                 const Eigen::ArrayXd &p_etypX,
                                 const Eigen::MatrixXd &p_svdMatrix,
                                 const bool    &p_bRotationAndRecale) :
        LocalLinearRegression(p_bZeroDate, p_nbMesh, p_mesh, p_mesh1D, p_meanX, m_etypX, p_svdMatrix, p_bRotationAndRecale)
    {
        m_particles = p_particles;
        m_simToCell = p_simToCell;
        m_matReg = p_matReg;
        m_diagReg = p_diagReg ;
        m_simulBelongingToCell = p_simulBelongingToCell;
    }
};
}
#endif /*LOCALLINEARREGRESSIONFORSDDP_H*/
