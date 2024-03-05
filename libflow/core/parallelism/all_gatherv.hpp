
//  Only partial mapping
#ifndef BOOST_MPI_ALL_GATHERV_HPP
#define BOOST_MPI_ALL_GATHERV_HPP
#include <boost/mpi/exception.hpp>
#include <boost/mpi/datatype.hpp>
#include <vector>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/assert.hpp>

namespace boost
{
namespace mpi
{

namespace detail
{
// We're gathering at the root for a type that has an associated MPI
// datatype, so we'll use MPI_All_Gatherv to do all of the work.
template<typename T>
void
all_gatherv_impl(const communicator &comm, const T *in_values, int in_size,
                 T *out_values, int *sizes, int *displs, mpl::true_)
{
    MPI_Datatype type = get_mpi_datatype<T>(*in_values);
    BOOST_MPI_CHECK_RESULT(MPI_Allgatherv,
                           (const_cast<T *>(in_values), in_size, type,
                            out_values, sizes, displs,   type,  comm));
}

} // end namespace detail


template<typename T, class V >
void
all_gatherv(const communicator &comm, const T *in_values, int in_size,
            V &out_values)
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
    if (out_values.size() != aux)
        out_values.resize(aux);

    detail::all_gatherv_impl(comm, in_values, in_size,
                             out_values.data(), &sizes[0], &displs[0],
                             is_mpi_datatype<T>());

}
template<typename T>
void
all_gatherv(const communicator &comm, const T *in_values, int in_size, T *out_values)
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
    detail::all_gatherv_impl(comm, in_values, in_size,
                             out_values, &sizes[0], &displs[0],
                             is_mpi_datatype<T>());

}
}
} // end namespace boost::mpi

#endif // BOOST_MPI_ALL_GATHERV_HPP
