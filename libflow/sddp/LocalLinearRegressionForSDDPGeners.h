
#ifndef LOCALLINEARREGRESSIONFORSDDPGENERS_H
#define LOCALLINEARREGRESSIONFORSDDPGENERS_H
#include <memory>
#include "libflow/sddp/LocalLinearRegressionForSDDP.h"
#include "geners/GenericIO.hh"
#include "geners/arrayIO.hh"
#include "libflow/core/utils/eigenGeners.h"

/** \file LocalLinearRegressionForSDDPGeners.h
 * \brief Define non intrusive  serialization with random acces
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(libflow::LocalLinearRegressionForSDDP, 1)
/// an external class
gs_declare_type_external(libflow::LocalLinearRegressionForSDDP)
///@}

namespace gs
{
//
// This is how the specialization of GenericWriter should look like
//
template <class Stream, class State>
struct GenericWriter < Stream, State, libflow::LocalLinearRegressionForSDDP,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const libflow::LocalLinearRegressionForSDDP &p_regression, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<libflow::LocalLinearRegressionForSDDP>());
        const bool status = p_processClassId ? current.write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            bool bZeroDate = p_regression.getBZeroDate();
            write_pod(p_os, bZeroDate);
            write_item(p_os, p_regression.getParticles());
            // copy to avoid duplicating eigen array mechanism
            write_item(p_os, p_regression.getNbMesh());
            write_item(p_os, p_regression.getMesh());
            write_item(p_os, p_regression.getMesh1D());
            write_item(p_os, p_regression.getSimToCell());
            write_item(p_os, p_regression.getMatReg());
            write_item(p_os, p_regression.getDiagReg());
            write_item(p_os, p_regression.getSimulBelongingToCell());
            write_item(p_os, p_regression.getMeanX());
            write_item(p_os, p_regression.getEtypX());
            write_item(p_os, p_regression.getSvdMatrix());
            write_pod(p_os, p_regression.getBRotationAndRescale());
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

// And this is the specialization of GenericReader
//
template <class Stream, class State>
struct GenericReader < Stream, State, libflow::LocalLinearRegressionForSDDP,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(libflow::LocalLinearRegressionForSDDP  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<libflow::LocalLinearRegressionForSDDP>());
        const ClassId &stored = p_processClassId ? ClassId(p_is, 1) : p_st->back();
        current.ensureSameId(stored);

        // Deserialize object data.
        bool bZeroDate = 0;
        read_pod(p_is, &bZeroDate);
        std::unique_ptr< Eigen::ArrayXXd>  particles = read_item< Eigen::ArrayXXd>(p_is);
        std::unique_ptr<Eigen::ArrayXi> nbMesh = gs::read_item< Eigen::ArrayXi>(p_is);
        std::unique_ptr<  Eigen::Array< std::array< double, 2>, Eigen::Dynamic, Eigen::Dynamic > > mesh = read_item<  Eigen::Array< std::array< double, 2>, Eigen::Dynamic, Eigen::Dynamic > >(p_is);
        std::unique_ptr<std::vector< std::shared_ptr< Eigen::ArrayXd > > > mesh1D = read_item<  std::vector< std::shared_ptr< Eigen::ArrayXd > > >(p_is);
        std::unique_ptr< Eigen::ArrayXi> simToCell = read_item<  Eigen::ArrayXi>(p_is);
        std::unique_ptr< Eigen::ArrayXXd > matReg = read_item< Eigen::ArrayXXd>(p_is);
        std::unique_ptr< Eigen::ArrayXXd > diagReg =  read_item< Eigen::ArrayXXd>(p_is);
        std::unique_ptr< std::vector<  std::shared_ptr< std::vector< int> > >  > simulBelongingToCell = read_item< std::vector<  std::shared_ptr< std::vector< int> > >  >(p_is);
        std::unique_ptr< Eigen::ArrayXd> meanX = read_item<  Eigen::ArrayXd>(p_is);
        std::unique_ptr< Eigen::ArrayXd> etypX = read_item<  Eigen::ArrayXd>(p_is);
        std::unique_ptr< Eigen::MatrixXd> svdMatrix = read_item<  Eigen::MatrixXd>(p_is);
        bool bRotationAndRescale = false;
        gs::read_pod(p_is, &bRotationAndRescale);
        if (p_is.fail())
            // Return "false" on failure
            return false;
        // Build the object from the stored data
        if (ptr)
        {
            *ptr = libflow::LocalLinearRegressionForSDDP(bZeroDate, *particles, *nbMesh, *mesh, *mesh1D, *simToCell, *matReg, *diagReg, *simulBelongingToCell, *meanX, *etypX, *svdMatrix, bRotationAndRescale);
        }
        else
        {
            ptr = new  libflow::LocalLinearRegressionForSDDP(bZeroDate, *particles, *nbMesh, *mesh, *mesh1D, *simToCell, *matReg, *diagReg, *simulBelongingToCell, *meanX, *etypX, *svdMatrix, bRotationAndRescale);
        }

        return true;
    }

    inline static bool process(libflow::LocalLinearRegressionForSDDP &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        libflow::LocalLinearRegressionForSDDP *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}

#endif /* LOCALLINEARREGRESSIONFORSDDPGENERS_H */
