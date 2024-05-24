
#ifdef _OPENMP
#ifndef OPENMPEXCEPTION_H
#define OPENMPEXCEPTION_H
#include <omp.h>
#include <mutex>

namespace reflow
{

/// \class OpenmpException
/// To generate and catch exceptions
class OpenmpException
{
    std::exception_ptr Ptr;
    std::mutex         Lock;
public:

    /// \brief Constructor
    OpenmpException(): Ptr(nullptr) {}
    /// \brief destructor
    ~OpenmpException()
    {
        this->rethrow();
    }

    /// \brief Rethrow
    void rethrow()
    {
        if (auto tmp = this->Ptr)
        {
            this->Ptr = nullptr;
            std::rethrow_exception(tmp);
        }
    }
    /// \brief Capture
    void captureException()
    {
        std::unique_lock<std::mutex> guard(this->Lock);
        this->Ptr = std::current_exception();
    }
    /// \brief Elegant way to run the test
    template <typename Function, typename... Parameters>
    void run(Function f, Parameters... params)
    {
        try
        {
            f(params...);
        }
        catch (...)
        {
            captureException();
        }
    }

};
}
#endif
#endif
