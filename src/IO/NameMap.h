/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C)        AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: : multiboost@googlegroups.com
 *
 *    For more information and up-to-date version, please visit
 *
 *                       http://www.multiboost.org/
 *
 */


/**
 * \file NameMap.h The map of the names of classes and enum type attributes.
 */

#ifndef __NAME_MAP_H
#define __NAME_MAP_H

#include <string>
#include <vector>
#include <map>
#include <iostream>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
 * A class to represent name (classes, enum attributes) map.
 * Each name is mapped into an index for efficiency reasons.
 * \date 14/11/2005
 */
    class NameMap
    {
    public:

        /**
         * The constructor. It does noting but initializing some variables.
         * \date 19/04/2007
         */
    NameMap() : _numRegNames(0) {  }

        /**
         * Add a name to the registered list.
         * \param name The name to add.
         * \date 19/04/2007
         */
        int addName(const string& name);

        /**
         * Get the name using the index.
         * Example:
         * \code
         * getNameFromIdx(0); // -> "rock"
         * \endcode
         * \param idx The index of the name.
         * \date 19/04/2007
         */
        string getNameFromIdx(int idx) const; 

        /**
         * Get the index using the name
         * Example:
         * \code
         * getIdxFromName("rock"); // -> 0
         * \endcode
         * \param name The name.
         * \date 19/04/2007
         */
        int getIdxFromName(const string& name) const;

        int getNumNames() const { return _numRegNames; }   //!< Returns the number of names 

        void clear( void );
    private:
        /**
         * Maps the internal index to the name: internal_index->name. 
         * \see getNameFromIdx
         * \date 19/04/2007
         */
        vector<string>   _mapIdxToName; 

        /**
         * Maps the name to the internal index: name->internal_index.
         * \see getIdxFromName
         * \date 19/04/2007
         */
        mutable map<string, int> _mapNameToIdx;

        int _numRegNames; //!< The number of the names registered. 
    };

} // end of namespace MultiBoost

#endif // NAME_MAP_H

