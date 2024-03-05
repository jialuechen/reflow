// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef _OPENMP
#ifndef OPENMPEXCEPTION_H
#define OPENMPEXCEPTION_H
#include <omp.h>
#include <mutex>

/** \file OPNEMPException.h
 * \brief Use to generate exception  in parallelized block with OPENMP
 *  \author Xavier Warin
 */

namespace libflow
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
