#ifndef _EIGENMEMORY_H
#define _EIGENMEMORY_H

#if EIGEN_ALIGN
void operator delete (void *ptr, std::size_t /* sz */) throw()
{
    Eigen::internal::conditional_aligned_free<NeedsToAlign>(ptr);
}
void operator delete[](void *ptr, std::size_t /* sz */) throw()
{
    Eigen::internal::conditional_aligned_free<NeedsToAlign>(ptr);
}
#endif
#endif /* _EIGENMEMORY_H */
