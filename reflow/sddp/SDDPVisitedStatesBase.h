
#ifndef SDDPVISITEDSTATESBASE_H
#define SDDPVISITEDSTATESBASE_H
#include <vector>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif

/**  \file SDDPVisitedStatesBase.h
 *   \brief Storing visited states during simulation
 *          All other SDDPVisited* classe derive from this class
 *   \author Xavier Warin
 */
namespace reflow
{
class SDDPVisitedStatesBase
{
protected :

    std::vector< std::shared_ptr< Eigen::ArrayXd > > m_stateVisited ; ///< vector of  state visited
    std::vector<int>  m_associatedMesh ; /// mesh associated (state visited conditionally) : this mesh corresponds to a point in tree or a domain set in uncertainty levels
    std::vector< std::vector< int> > m_meshToState ; /// To a  node (tree) or mesh (regression)  number j associates all m_stateVisited  and m_associatedMesh   index  i such that m_associatedMesh[i]= j

    /// \brief Check is a state is already stored
    ///        Should be useful for Bang Bang
    /// \param  p_state state to add potentially
    /// \param  p_point number of the node in the tree or mesh number  at a given date where the state should  added
    bool isStateNotAlreadyVisited(const std::shared_ptr< Eigen::ArrayXd > &p_state,  const int  &p_point) const ;

    /// \brief  Eliminate doubling states
    void recalculateVisitedState();


public:

    /// \brief Default constructor
    SDDPVisitedStatesBase();

    /// \param p_nbNode  number of nodes or mesh  for uncertainties
    SDDPVisitedStatesBase(const int &p_nbNode);

    /// \brief Second constructor with all states
    SDDPVisitedStatesBase(const std::vector< std::vector< int> >   &p_meshToState, const std::vector< std::shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const std::vector<int> &p_associatedMesh) ;


    ///\brief Some accessor
    //@{
    inline  int getStateSize() const
    {
        return m_stateVisited.size();   /// Number of states visited
    }
    inline int getMeshAssociatedToState(const int &p_istate) const
    {
        return  m_associatedMesh[p_istate];
    }
    inline  std::shared_ptr<Eigen::ArrayXd > getAState(const int &p_istate) const
    {
        return m_stateVisited[p_istate];
    }
    inline  const std::vector< std::shared_ptr< Eigen::ArrayXd > > &getStateVisited() const
    {
        return  m_stateVisited;
    }
    inline  const std::vector<int>  &getAssociatedMesh() const
    {
        return m_associatedMesh ;
    }
    inline const std::vector< std::vector< int> > &getMeshToState() const
    {
        return m_meshToState;
    }
    //@}

    ///\brief print function for debug
    void print() const;

#ifdef USE_MPI
    /// \brief Send all cut to root
    void sendToRoot(const boost::mpi::communicator &p_world);

    /// \brief Send all cuts from root to all processor
    void sendFromRoot(const boost::mpi::communicator &p_world);
#endif

};

}
#endif /* SDDPVISITEDSTATESBASE_H */
