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


#ifndef __SCALARLEARNER_H_
#define __SCALARLEARNER_H_

#include "BaseLearner.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
 * \file ScalarLearner.h It represents all weak learners that divide into two parts
 * of data using the implementation of phi(x,l) independently on the feature idx.
 *
 * \date 19/07/10
 */

    class ScalarLearner: public virtual BaseLearner
    {
    public:
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of
         * the object.
         */
        virtual ~ScalarLearner() {}

        /**
         * The single cut function for a particular feature. This function can return
         * with zero as well, indicating that the classifier abstained to classify
         * the instance having index idx.
         * \return +1,-1,0
         * \date 21/10/2010       
         */
        virtual AlphaReal cut( InputData* pData, int idx ) const = 0;
    };

}
#endif /* SCALARLEARNER_H_ */
