// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef STDSHAREDPTRSERIALIZATION_H
#define STDSHAREDPTRSERIALIZATION_H
#include <boost/serialization/split_free.hpp>
#include <boost/unordered_map.hpp>

//---/ Wrapper for std::shared_ptr<> /------------------------------------------

namespace boost
{
namespace serialization
{

template<class Archive, class Type>
void save(Archive &archive, const std::shared_ptr<Type> &value, const unsigned int /*version*/)
{
    Type *data = value.get();
    archive << data;
}

template<class Archive, class Type>
void load(Archive &archive, std::shared_ptr<Type> &value, const unsigned int /*version*/)
{
    Type *data;
    archive >> data;

    typedef std::weak_ptr<Type> WeakPtr;
    static boost::unordered_map<void *, WeakPtr> hash;

    if (hash[data].expired())
    {
        value = std::shared_ptr<Type>(data);
        hash[data] = value;
    }
    else value = hash[data].lock();
}

template<class Archive, class Type>
inline void serialize(Archive &archive, std::shared_ptr<Type> &value, const unsigned int version)
{
    split_free(archive, value, version);
}

}
}
#endif
