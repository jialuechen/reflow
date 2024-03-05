
#ifndef VERSION_H
#define VERSION_H
#define libflow_VERSION "5.11"
#include <string>

/** \file version.h
 * \brief Defines libflow version
 * \author Xavier Warin
 */

namespace libflow
{
/// \brief get back library version
std::string getlibflowVersion();
}
#endif
