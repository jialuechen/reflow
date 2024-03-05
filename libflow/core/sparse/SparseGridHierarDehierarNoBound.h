// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SPARSEGRIDHIERARDEHIERARNOBOUND_H
#define  SPARSEGRIDHIERARDEHIERARNOBOUND_H
#include  <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/SparseGridHierarDehierar.h"

/** \file SparseGridHierarDehierarNoBound.h
 * \brief Specialization for Hierarchization , Dehierarchization class for sparse grids eliminating boundary points
 * \author Xavier Warin
 */

namespace libflow
{

/// \class HierarDehierarNoBound SparseGridHierarDehierarNoBound.h
///  Abstract class for Hierarchization and Dehierarchization
class HierarDehierarNoBound : public HierarDehierar
{
public :

    /// \brief Default constructor
    HierarDehierarNoBound() {}

    /// \brief Get root point
    /// \param p_levelRoot     root level
    /// \param p_positionRoot  root position
    void  get_root(Eigen::ArrayXc &p_levelRoot, Eigen::ArrayXui   &p_positionRoot)
    {
        p_levelRoot.setConstant(1);
        p_positionRoot.setConstant(0);
    }
};
}
#endif /*  SPARSEGRIDHIERARDEHIERARNOBOUND_H */
