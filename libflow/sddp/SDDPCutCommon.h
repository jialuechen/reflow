
#ifndef SDDPCUTCOMMON_H
#define SDDPCUTCOMMON_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <array>
#include "geners/BinaryFileArchive.hh"
#include "libflow/sddp/SDDPACut.h"

/** \file SDDPCutCommon.h
 *  \brief utilities to deal with cuts under mpi
 *
 *  \author Xavier Warin
 */

namespace libflow
{
class SDDPCutCommon
{
public:

    // default constructor
    SDDPCutCommon() {}


protected :

#ifdef USE_MPI

    /// \brief Permits to calculate the routing schedule
    ///  cuts per simulations are spread over processors : they are reorganized in order to ease conditional expectation on all processors available
    /// \param   p_nbLP                 Total number of LP to solve
    /// \param   p_nbPartPerState       Number of LP to solve for each state visited
    /// \param   p_tabSendToProcessor   gives LP to send to another processor
    /// \param   p_tabRecFromProcessor  gives LP to receive from another processor
    /// \param   p_stateByProc          gives the state vector resolved by current processor
    /// \param   p_world                MPI communicator
    /// \return  number of LP solved by current processor
    void routingSchedule(const int &p_nbLP, const std::vector<int>   &p_nbPartPerState,
                         std::vector< std::array<int, 2> > &p_tabSendToProcessor,
                         std::vector< std::array<int, 2> > &p_tabRecFromProcessor,
                         std::array<int, 2>    &p_stateByProc,
                         const boost::mpi::communicator &p_world
                        ) const ;



    /// \brief mpi communications
    /// \param    p_cutPerSim           cut solved by current processor
    /// \param    p_tabSendToProcessor   gives LP to send to another processor
    /// \param    p_tabRecFromProcessor  gives LP to receive from another processor
    /// \param    p_cutPerSimProc       gives cut  necessary for current processor to calculate conditional expectation
    /// \param   p_world                MPI communicator
    void mpiExec(const Eigen::ArrayXXd   &p_cutPerSim,
                 std::vector< std::array<int, 2> > &p_tabSendToProcessor,
                 std::vector< std::array<int, 2> > &p_tabRecFromProcessor,
                 Eigen::ArrayXXd    &p_cutPerSimProc,
                 const boost::mpi::communicator &p_world) const ;


    /// \brief After all processors have calculated some conditional expectation of the cuts, these cuts are shared between all processors
    /// \param p_localCut  Vector of conditional cuts calculated by the current processor / in output all the cuts of the problem
    /// \param p_meshCut for each conditional cut gives the mesh where it is located    (entry for the processors, output for all processors)
    /// \param   p_world                MPI communicator
    void mpiExecCutRoutage(std::vector< std::shared_ptr<SDDPACut> > &p_localCut,
                           std::vector< int > &p_meshCut,
                           const boost::mpi::communicator &p_world);
#endif



    /// \brief Load already calculated cuts
    /// \param p_ar    archive to load cuts
    /// \param p_name  base name for cuts
    /// \param p_node  number of mesh or nodes at current time step
    /// \param p_date  date number
    /// \param p_cuts  set of cuts by node
    /// \param p_world  MPI communicator
    void loadCutsByName(const std::shared_ptr<gs::BinaryFileArchive> &p_ar, const std::string &p_name, const int &p_node, const int &p_date,
                        std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &p_cuts
#ifdef USE_MPI
                        , const boost::mpi::communicator &p_world
#endif

                       );


    /// \brief During  cut creation :
    ///        Gather all cuts  and store them
    ///  \param p_localCut   all new cuts to store
    ///  \param p_nodeCut    for each cut, the node or mesh involved
    ///  \param p_name       name of the cuts in binary archive
    ///  \param p_cuts       cuts updated
    ///  \param p_ar         binary archive used to store cuts
    ///  \param p_nbNodes    number of nodes or meshes
    ///  \param p_date       date index
    ///  \param p_world  MPI communicator
    void gatherAndStoreCuts(std::vector< std::shared_ptr<SDDPACut> > &p_localCut,  std::vector<int> &p_nodeCut, const std::string &p_name, std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &p_cuts, const std::shared_ptr<gs::BinaryFileArchive> &p_ar, const int  &p_nbNodes, const int   &p_date
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           );

};
}
#endif
