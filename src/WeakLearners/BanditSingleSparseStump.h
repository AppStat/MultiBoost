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
* \file BanditSingleSparseStump.h A single threshold decision stump learner. 
*/

#ifndef __BANDIT_SINGLE_SPARSE_STUMP_H
#define __BANDIT_SINGLE_SPARSE_STUMP_H

//#include "WeakLearners/ClasswiseLearner.h"
#include "WeakLearners/FeaturewiseLearner.h"
#include "WeakLearners/BanditSingleStumpLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"
#include "IO/SortedData.h"
#include "Bandits/Exp3G2.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* A \b single threshold decision stump learner. 
* There is ONE and ONE ONLY threshold here.
*/
class BanditSingleSparseStump : public BanditSingleStumpLearner
{
public:
	BanditSingleSparseStump() : BanditSingleStumpLearner() {}

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~BanditSingleSparseStump() {}

   /**
   */
   virtual void init();


   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { 
	   BaseLearner* retLearner = new BanditSingleSparseStump();
	   dynamic_cast< BanditSingleStumpLearner* >(retLearner)->setBanditAlgoObject( _banditAlgo );
	   return retLearner;
   }

   /**
   * Run the learner to build the classifier on the given data.
   * \param pData The pointer to the data.
   * \see BaseLearner::run
   * \date 11/11/2005
   */
   virtual AlphaReal run();

   virtual AlphaReal run( int colIdx );

protected:

};

//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __SINGLE_STUMP_LEARNER_H
