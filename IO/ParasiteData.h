/*
* This file is part of MultiBoost, a multi-class 
* AdaBoost learner/classifier
*
* Copyright (C) 2005-2006 Norman Casagrande
* For informations write to nova77@gmail.com
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
*/

/**
* \file ParasiteData.h Input data which has the column sorted.
*/

#ifndef __PARASITE_DATA_H
#define __PARASITE_DATA_H

#include "InputData.h"
#include "WeakLearners/BaseLearner.h"

#include <vector>

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
* Overloading of the InputData class to support ParasiteLearner.
* it stores a pool of baselearners already learned
* \date 24/04/2007
*/
class ParasiteData : public InputData
{
public:

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~ParasiteData() {}

   int getNumBaseLearners() const 
   { return _baseLearners.size(); }   //!< Returns the number of base learners

   BaseLearner getBaseLearner(int i) const 
   { return _baseLearners[i]; }   //!< Returns the number of base learners

   void addBaseLearner(const BaseLearner& baseLearner) 
   { _baseLearners.push_back(baseLearner); }   //!< Returns the number of base learners

protected:

   vector<BaseLearner>    _baseLearners; //!< the pool of base learners

};

} // end of namespace MultiBoost

#endif // __PARASITE_DATA_H
