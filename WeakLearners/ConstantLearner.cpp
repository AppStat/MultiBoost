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

#include "ConstantLearner.h"

#include "IO/Serialization.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>

namespace MultiBoost {


//REGISTER_LEARNER_NAME(Constant, ConstantLearner)
REGISTER_LEARNER(ConstantLearner)

// ------------------------------------------------------------------------------

float ConstantLearner::run()
{

   const int numClasses = _pTrainingData->getNumClasses();
   //const int numColumns = _pTrainingData->getNumColumns();

   // set the smoothing value to avoid numerical problem
   // when theta=0.
   setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

   vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
   vector<float> tmpV(numClasses); // The class-wise votes/abstentions

   ConstantAlgorithm cAlgo;
   cAlgo.findConstant(_pTrainingData,&mu,&tmpV);
   
   _v = tmpV;

   return getEnergy(mu, _alpha, _v);
}

// ------------------------------------------------------------------------------

float ConstantLearner::run( int colNum )
{
	return this->run();
}

// -----------------------------------------------------------------------

void ConstantLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
{
//    const int numClasses = ClassMappings::getNumClasses();
//    const int numExamples = pData->getNumExamples();

//    // reason ignored for the moment as it is used for a single task
//    data.resize( numClasses + numExamples );

//    int pos = 0;

//    for (int l = 0; l < numClasses; ++l)
//       data[pos++] = _v[l];

//    for (int i = 0; i < numExamples; ++i)
//       data[pos++] = ConstantLearner::phi( pData->getValue( i, _selectedColumn), 0 );
}

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
