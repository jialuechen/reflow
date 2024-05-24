
#ifndef SPARSEGRIDHIERARDEHIERAR_H
#define  SPARSEGRIDHIERARDEHIERAR_H
#include  <Eigen/Dense>
#include "reflow/core/sparse/sparseGridTypes.h"


namespace reflow
{

/// \class HierarDehierar SparseGridHierarDehierar.h
///  Abstract class for Hierarchization and Dehierarchization
class HierarDehierar
{
public :

    /// \brief Default constructor
    HierarDehierar() {}

    /// \brief Default destructor
    virtual ~HierarDehierar() {}

    /// \brief Get root point
    /// \param p_levelRoot     root level
    /// \param p_positionRoot  root position
    virtual void  get_root(Eigen::ArrayXc &p_levelRoot, Eigen::ArrayXui   &p_positionRoot)  = 0;


};
}
#endif /*  SPARSEGRIDHIERARDEHIERAR_H */
