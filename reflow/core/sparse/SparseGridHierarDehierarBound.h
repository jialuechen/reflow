
#ifndef SPARSEGRIDHIERARDEHIERARBOUND_H
#define  SPARSEGRIDHIERARDEHIERARBOUND_H
#include  <Eigen/Dense>
#include "reflow/core/sparse/sparseGridTypes.h"
#include "reflow/core/sparse/SparseGridHierarDehierar.h"

/** \file SparseGridHierarDehierarNoBound.h
 * \brief Specialization for Hierarchization , Dehierarchization class for sparse grids with boundary points
 * \author Xavier Warin
 */

namespace reflow
{

/// \class HierarDehierarBound SparseGridHierarDehierarBound.h
///  Abstract class for Hierarchization and Dehierarchization
class HierarDehierarBound : public HierarDehierar
{
public :

    /// \brief Default constructor
    HierarDehierarBound() {}

    /// \brief Get root point
    /// \param p_levelRoot     root level
    /// \param p_positionRoot  root position
    void  get_root(Eigen::ArrayXc &p_levelRoot, Eigen::ArrayXui   &p_positionRoot)
    {
        p_levelRoot.setConstant(1);
        p_positionRoot.setConstant(1);
    }
};
}
#endif /*  SPARSEGRIDHIERARDEHIERARBOUND_H */
