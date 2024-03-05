
//  Only partial mapping
#ifndef MY_BOOST_MPI_GATHERV_HPP
#define MY_BOOST_MPI_GATHERV_HPP
#include <boost/version.hpp>
#include <boost/mpi/exception.hpp>
#include <boost/mpi/datatype.hpp>
#include <vector>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/assert.hpp>
#if BOOST_VERSION  >= 105900
#include <boost/mpi/collectives/gatherv.hpp>
#endif

namespace boost
{
namespace mpi
{
#if BOOST_VERSION< 105900
namespace detail
{
// We're gathering at the root for a type that has an associated MPI
// datatype, so we'll use MPI_Gatherv to do all of the work.
template<typename T>
void
gatherv_impl(const communicator &comm, const T *in_values, int in_size,
             T *out_values, int *sizes, int *displs, int root,  mpl::true_)
{
    MPI_Datatype type = get_mpi_datatype<T>(*in_values);
    BOOST_MPI_CHECK_RESULT(MPI_Gatherv,
                           (const_cast<T *>(in_values), in_size, type,
                            out_values, sizes, displs,   type, root, comm));
}

} // end namespace detail
#endif

template<typename T, class V >
void
gatherv(const communicator &comm, const T *in_values, int in_size,
        V &out_values, int root)
{
    int nprocs = comm.size();

    std::vector<int> sizes(nprocs);
    ::boost::mpi::all_gather(comm, in_size, sizes);

    std::vector<int> displs(nprocs);
    int aux = 0 ;
    for (int rank = 0; rank < nprocs; ++rank)
    {
        displs[rank] = aux;
        aux += sizes[rank];
    }
    if (comm.rank() == root)
    {
        if (out_values.size() != aux)
            out_values.resize(aux);
    }
    detail::gatherv_impl(comm, in_values, in_size,
                         out_values.data(), &sizes[0], &displs[0], root,
                         is_mpi_datatype<T>());

}

}
} // end namespace boost::mpi

#endif // MY_BOOST_MPI_GATHERV_HPP

