
#include "libflow/sddp/SDDPCutCommon.h"
#include "boost/lexical_cast.hpp"
#ifdef USE_MPI
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/sddp/SDDPACutGeners.h"


using namespace Eigen ;
using namespace std;
using namespace gs;

namespace libflow
{

#ifdef USE_MPI
void  SDDPCutCommon::routingSchedule(const int &p_nbLP, const vector<int>   &p_nbPartPerState,
                                     vector< array<int, 2> > &p_tabSendToProcessor,
                                     vector< array<int, 2> > &p_tabRecFromProcessor,
                                     array<int, 2>   &p_stateByProc,
                                     const boost::mpi::communicator &p_world) const
{
    vector< array< int, 2> >  lpBelongingToProc(p_world.size());
    int nsimPProc = (int)(p_nbLP / p_world.size());
    int nRest = p_nbLP % p_world.size();
    for (int ip = 0 ; ip < p_world.size(); ++ip)
    {
        if (ip < nRest)
        {
            lpBelongingToProc[ip][0] = ip * nsimPProc + ip;
            lpBelongingToProc[ip][1] = lpBelongingToProc[ip][0] + nsimPProc + 1;
        }
        else
        {
            lpBelongingToProc[ip][0] = ip * nsimPProc + nRest;
            lpBelongingToProc[ip][1] = lpBelongingToProc[ip][0] + nsimPProc;
        }
    }
    vector< array<int, 2> >  lpNeededByProc(p_world.size());
    int nbState = p_nbPartPerState.size();
    int nbStateProc = (int)(nbState / p_world.size());
    int nRestState = nbState % p_world.size();
    // calculate the LP simulations number (first and last (excluded)) needed by each processor to calculate conditional expectation
    int lpCounter = 0 ;
    for (int  ip = 0 ; ip < p_world.size(); ++ip)
    {
        lpNeededByProc[ip][0] = lpCounter;
        // conditional expectation number
        int iStateMin ;
        int iStateMax ;
        if (ip < nRestState)
        {
            iStateMin = ip * nbStateProc + ip;
            iStateMax = iStateMin + nbStateProc + 1;
        }
        else
        {
            iStateMin = ip * nbStateProc + nRestState;
            iStateMax = iStateMin + nbStateProc;
        }
        // traduction in term of lp
        for (int iProc = iStateMin; iProc < iStateMax ; ++iProc)
        {
            lpCounter += p_nbPartPerState[iProc];
        }
        lpNeededByProc[ip][1] = lpCounter;
        if (ip == p_world.rank())
        {
            p_stateByProc[0] = iStateMin;
            p_stateByProc[1] = iStateMax ;
        }
    }

    /// Particule to send to another processor (processor number)(part min, part max +1)
    p_tabSendToProcessor.resize(p_world.size());
    for (int ip = 0 ; ip < p_world.size() ; ++ip)
    {
        size_t iPartMin = max(lpNeededByProc[ip][0], lpBelongingToProc[p_world.rank()][0]);
        size_t iPartMax  = min(lpNeededByProc[ip][1], lpBelongingToProc[p_world.rank()][1]);
        if (iPartMin < iPartMax)
        {
            // in itask processor referential
            p_tabSendToProcessor[ip][0] = iPartMin - lpBelongingToProc[p_world.rank()][0];
            p_tabSendToProcessor[ip][1] = iPartMax - lpBelongingToProc[p_world.rank()][0];
        }
        else
        {
            p_tabSendToProcessor[ip][0] = -1;
            p_tabSendToProcessor[ip][1] = -1;
        }
    }
    for (int ip = p_world.size() ; ip < p_world.size() ; ++ip)
    {
        p_tabSendToProcessor[ip][0] = -1;
        p_tabSendToProcessor[ip][1] = -1;
    }

    /// Particule to receive from another processor (processor number)(part min, part max +1)
    p_tabRecFromProcessor.resize(p_world.size());
    for (int ip = 0 ; ip < p_world.size() ; ++ip)
    {
        size_t iPartMin = max(lpNeededByProc[p_world.rank()][0], lpBelongingToProc[ip][0]);
        size_t iPartMax  = min(lpNeededByProc[p_world.rank()][1], lpBelongingToProc[ip][1]);
        if (iPartMin < iPartMax)
        {
            // in ip processor referential
            p_tabRecFromProcessor[ip][0] = iPartMin - lpNeededByProc[p_world.rank()][0];
            p_tabRecFromProcessor[ip][1] = iPartMax - lpNeededByProc[p_world.rank()][0];
        }
        else
        {
            p_tabRecFromProcessor[ip][0] = -1;
            p_tabRecFromProcessor[ip][1] = -1;
        }
    }
}


void SDDPCutCommon::mpiExec(const ArrayXXd   &p_cutPerSim,
                            vector< array<int, 2> > &p_tabSendToProcessor,
                            vector< array<int, 2> > &p_tabRecFromProcessor,
                            ArrayXXd    &p_cutPerSimProc,
                            const boost::mpi::communicator &p_world) const
{
    // number of rosws
    int nbRows = p_cutPerSim.rows();
    // communication receive part
    vector<  boost::mpi::request > reqRec(p_world.size());
    int nbRec = 0 ;
    // communication send part
    for (int iproc = 0 ; iproc < p_world.size(); ++iproc)
    {
        // size of send
        int sizeRec = (p_tabRecFromProcessor[iproc][1] - p_tabRecFromProcessor[iproc][0]) * nbRows;
        if (iproc != p_world.rank())
        {
            if (sizeRec > 0)
            {
                // communication
                int imesg_iproc = 0;
                reqRec[nbRec++] = p_world.irecv(iproc, imesg_iproc++, p_cutPerSimProc.col(p_tabRecFromProcessor[iproc][0]).data(), sizeRec);
            }
        }
        else
        {
            // same processor
            if (p_tabRecFromProcessor[p_world.rank()][1] > p_tabRecFromProcessor[p_world.rank()][0])
            {
                int begBlockRec = p_tabRecFromProcessor[p_world.rank()][0];
                int sizeBlockRec = p_tabRecFromProcessor[p_world.rank()][1] - begBlockRec;
                int begBlockSend = p_tabSendToProcessor[p_world.rank()][0];
                int sizeBlockSend = p_tabSendToProcessor[p_world.rank()][1] - begBlockSend;
                assert(sizeBlockRec == sizeBlockSend);
                p_cutPerSimProc.block(0, begBlockRec, nbRows, sizeBlockRec)  =  p_cutPerSim.block(0, begBlockSend, nbRows, sizeBlockSend);
            }
        }
    }
    vector<   boost::mpi::request > reqSend(p_world.size());
    int nbSend = 0 ;
    // communication send part
    for (int iproc = 0 ; iproc < p_world.size(); ++iproc)
    {
        // size of send
        int sizeSend = (p_tabSendToProcessor[iproc][1] - p_tabSendToProcessor[iproc][0]) * nbRows;
        if ((sizeSend > 0) && (iproc != p_world.rank()))
        {
            int imesg_iproc = 0;
            reqSend[nbSend++] =  p_world.isend(iproc, imesg_iproc++, p_cutPerSim.col(p_tabSendToProcessor[iproc][0]).data(), sizeSend);
        }
    }
    boost::mpi::wait_all(reqRec.begin(), reqRec.begin() + nbRec);
    boost::mpi::wait_all(reqSend.begin(), reqSend.begin() + nbSend);
}


void SDDPCutCommon::mpiExecCutRoutage(vector< shared_ptr<SDDPACut> > &p_localCut, vector< int > &p_meshCut,
                                      const boost::mpi::communicator &p_world)
{
    int isizeCut  = 0;
    int iRowCut   = 0;
    int iColCut   = 0 ;
    if (p_world.rank() == 0)
    {
        isizeCut =  p_localCut[0]->getCut()->size();
        iRowCut =   p_localCut[0]->getCut()->rows();
        iColCut  =  p_localCut[0]->getCut()->cols();
    }
    broadcast(p_world, isizeCut, 0);
    broadcast(p_world, iRowCut, 0);
    broadcast(p_world, iColCut, 0);
    ArrayXd arrayToGather(p_localCut.size()*isizeCut);
    for (size_t iCut = 0; iCut < p_localCut.size(); ++iCut)
    {
        Map< const ArrayXd > tab(p_localCut[iCut]->getCut()->data(), isizeCut);
        arrayToGather.segment(iCut * isizeCut, isizeCut) = tab;
    }
    // global vector
    ArrayXd  allCuts ;
    all_gatherv(p_world, arrayToGather.data(), arrayToGather.size(), allCuts);
    vector<int> allMesh;
    all_gatherv(p_world, p_meshCut.data(), p_meshCut.size(), allMesh);
    // create the cuts
    p_meshCut = allMesh;
    int nbTotCuts = allCuts.size() / isizeCut;
    p_localCut.resize(nbTotCuts);
    for (int i = 0; i < nbTotCuts; ++i)
    {
        shared_ptr<ArrayXXd > ptCut = make_shared< ArrayXXd>(iRowCut, iColCut);
        Map<ArrayXXd> mapCut(allCuts.data() + i * isizeCut, iRowCut, iColCut);
        *ptCut = mapCut;
        p_localCut[i] = make_shared< SDDPACut>(ptCut);
    }
}
#endif

void SDDPCutCommon::loadCutsByName(const shared_ptr< BinaryFileArchive>   &p_ar, const std::string &p_name, const int &p_node, const int &p_date,   std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &p_cuts
#ifdef USE_MPI
                                   , const boost::mpi::communicator &p_world
#endif
                                  )
{

#ifdef USE_MPI
    int itask = p_world.rank();
    if (itask == 0)
    {
#endif
        string stringStep = boost::lexical_cast<string>(p_date);
        p_cuts.resize(p_node);
        for (int i = 0; i < p_node; ++i)
        {
            string stringCutMesh = p_name + boost::lexical_cast<string>(i);
            // number of cuts already generated
            Reference< SDDPACut > refCut(*p_ar, stringCutMesh, stringStep);
            p_cuts[i].resize(refCut.size());
            for (size_t j = 0; j < refCut.size(); ++j)
                p_cuts[i][j] = refCut.getShared(j);
        }

#ifdef USE_MPI
    }
    // use mpi to spread cuts
#if BOOST_VERSION <  105600
    boost::mpi::broadcast(p_world, p_cuts, 0);
#else
    // boost bug ?
    vector<int> ivec;
    vector< ArrayXXd> tabSerial;
    if (itask == 0)
    {
        ivec.resize(p_cuts.size());
        int itotal = 0 ;
        for (size_t i = 0; i < ivec.size(); ++i)
        {
            ivec[i] = p_cuts[i].size();
            itotal += p_cuts[i].size();
        }
        tabSerial.reserve(itotal);
        for (size_t i = 0; i < ivec.size(); ++i)
            for (int j = 0 ; j < ivec[i]; ++j)
                tabSerial.push_back(*p_cuts[i][j]->getCut());
    }
    boost::mpi::broadcast(p_world, ivec, 0);
    boost::mpi::broadcast(p_world, tabSerial, 0);
    if (itask > 0)
    {
        p_cuts.resize(ivec.size());
        int ipos = 0;
        for (size_t i = 0; i < ivec.size(); ++i)
        {
            p_cuts[i].resize(ivec[i]);
            for (int j = 0 ; j < ivec[i]; ++j)
            {
                shared_ptr<ArrayXXd> ptrTab = make_shared<ArrayXXd>(ArrayXXd(tabSerial[ipos++]));
                p_cuts[i][j] = make_shared<SDDPACut>(SDDPACut(ptrTab));
            }
        }
    }

#endif

#endif

}

void SDDPCutCommon::gatherAndStoreCuts(vector< shared_ptr<SDDPACut> > &p_localCut,   vector<int> &p_nodeCut, const string &p_name, vector< vector<  shared_ptr<SDDPACut> > > &p_cuts, const shared_ptr<BinaryFileArchive> &p_ar, const int &p_nbNodes, const int   &p_date
#ifdef USE_MPI
                                       , const boost::mpi::communicator &p_world
#endif
                                      )
{


#ifdef USE_MPI
    mpiExecCutRoutage(p_localCut, p_nodeCut, p_world);
#endif
    vector<int> nbCutPerNode(p_nbNodes, 0);
    for (size_t i = 0; i < p_nodeCut.size(); ++i)
        nbCutPerNode[p_nodeCut[i]] += 1;
    // add  cuts
    vector< vector<  shared_ptr<SDDPACut> > > additionalCuts(nbCutPerNode.size());
    for (int inode = 0; inode < p_nbNodes; ++inode)
        additionalCuts[inode].reserve(nbCutPerNode[inode]);
    for (size_t i = 0; i < p_nodeCut.size(); ++i)
    {
        additionalCuts[p_nodeCut[i]].push_back(p_localCut[i]);
    }
    // now store additional cuts
#ifdef USE_MPI
    int itask = p_world.rank();

#else
    int itask = 0;
#endif
    if (itask == 0)
    {
        string stringStep = boost::lexical_cast<string>(p_date);
        for (size_t i = 0; i < additionalCuts.size(); ++i)
        {
            if (additionalCuts[i].size() > 0)
            {
                string stringCutNode = p_name + boost::lexical_cast<string>(i);
                for (size_t j = 0; j < additionalCuts[i].size(); ++j)
                {
                    *p_ar << Record(*additionalCuts[i][j], stringCutNode, stringStep);
                }
            }
        }
    }
    // add additional cuts to cuts already present for next (backward) time step
    for (size_t i = 0; i < additionalCuts.size(); ++i)
    {
        if (additionalCuts[i].size() > 0)
        {
            int isize = p_cuts[i].size();
            p_cuts[i].resize(isize + additionalCuts[i].size());
            for (size_t j = 0; j < additionalCuts[i].size(); ++j)
            {
                p_cuts[i][isize + j] = additionalCuts[i][j];
            }
        }
    }
}


}
