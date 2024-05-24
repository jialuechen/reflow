
#include <boost/version.hpp>
#include <iostream>
#include <Eigen/Core>
#include "reflow/core/utils/version.h"

using namespace std ;

namespace reflow
{
std::string getreflowVersion()
{
    std::stringstream reflow_version;

    reflow_version << "reflow version "
                  << reflow_VERSION ;

    reflow_version << ";  Boost version " << BOOST_VERSION / 100000 << "." << BOOST_VERSION / 100 % 1000 << "." << BOOST_VERSION % 100 << endl;

    reflow_version << ";  Eigen version 3." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION ;

    return reflow_version.str();
}
}
