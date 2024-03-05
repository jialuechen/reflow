
#include <boost/version.hpp>
#include <iostream>
#include <Eigen/Core>
#include "libflow/core/utils/version.h"

using namespace std ;

namespace libflow
{
std::string getlibflowVersion()
{
    std::stringstream libflow_version;

    libflow_version << "libflow version "
                  << libflow_VERSION ;

    libflow_version << ";  Boost version " << BOOST_VERSION / 100000 << "." << BOOST_VERSION / 100 % 1000 << "." << BOOST_VERSION % 100 << endl;

    libflow_version << ";  Eigen version 3." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION ;

    return libflow_version.str();
}
}
