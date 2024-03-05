
#ifndef PARALLELHITER_H
#define PARALLELHITER_H
#include <array>
#include <Eigen/Dense>

namespace libflow
{
/// \file  ParallelHiter.h
/// \class ParallelHiter ParallelHiter.h
/// \brief This iterator permits to iterate on each segment of an HyperCube (Cube), permitting to get
///  data inside an HCube and copying them before sending them to another processor
class ParallelHiter
{

private :

    bool m_bValid;

    Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >  m_hBounds; ///< Intersection hyperCube
    Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >  m_hBoundsGlob; ///< Global hypercube containing the previous one
    Eigen::ArrayXi  m_hVal; ///< Utilitarian to store current position of the iterator
    Eigen::ArrayXi  m_strides ; ///<  strides associated to global HyperCube

public :

    /// \brief Construction
    /// \param p_hBounds      hypercube bounds
    /// \param p_hBoundsGlob  hypercube bounds containing the previous one
    ParallelHiter(const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > >    &p_hBounds,
                  const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > >    &p_hBoundsGlob);

    ~ParallelHiter()
    {}


    /// \brief  Get the current position of the HIterator
    inline int  get() const
    {
        int iposInCube = 0 ;
        for (int idim = 0  ; idim <  m_hBounds.size(); ++idim)
        {
            iposInCube += (m_hVal(idim) - m_hBoundsGlob(idim)[0]) * m_strides(idim);
        }
        return iposInCube;
    }

    /// \brief  Check if the hiterator is valid: i.e.: if it has bypassed its upper bound
    inline bool isValid(void) const
    {
        return m_bValid ;
    }

    /// \brief  Get the size of the first dim segment and of the entire indexed hcube
    inline int segmentSize() const
    {
        return m_hBounds(0)[1] - m_hBounds(0)[0];
    }

    /// \brief number of points in the hypercube
    size_t hCubeSize();


    /// \brief  Go to the next value of the hcube-iterator  : increase (+1) the hcube-iterator (from dim 2 (lower) to N (upper)
    bool next(void);

};
}
#endif // PARALLEL_HITER 
