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

#include "DirichletSingleStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"
#include "Algorithms/StumpAlgorithm.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace MultiBoost {

//REGISTER_LEARNER_NAME(SingleStump, DirichletSingleStumpLearner)
REGISTER_LEARNER(DirichletSingleStumpLearner)

vector< float > DirichletSingleStumpLearner::_featurewiseMin; 
vector< float > DirichletSingleStumpLearner::_featurewiseMax; 
int DirichletSingleStumpLearner::_numOfCalling = 0; //number of the single stump learner had been called

//-------------------------------------------------------------------------------

void DirichletSingleStumpLearner::init() {
	DirichletSingleStumpLearner::_numOfCalling = 1;

	DirichletSingleStumpLearner::_featurewiseMax.resize( _pTrainingData->getNumAttributes() );
	DirichletSingleStumpLearner::_featurewiseMin.resize( _pTrainingData->getNumAttributes() );
	
	//_pTrainingData->

    const int numClasses = _pTrainingData->getNumClasses();
    const int numColumns = _pTrainingData->getNumAttributes();

	
}


// ------------------------------------------------------------------------------

float DirichletSingleStumpLearner::run()
{
   DirichletSingleStumpLearner::_numOfCalling++;

   const int numClasses = _pTrainingData->getNumClasses();
   const int numColumns = _pTrainingData->getNumAttributes();

   // set the smoothing value to avoid numerical problem
   // when theta=0.
   setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

   vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
   vector<float> tmpV(numClasses); // The class-wise votes/abstentions

   float tmpThreshold;
   float tmpAlpha;

   float bestEnergy = numeric_limits<float>::max();
   float tmpEnergy;

   StumpAlgorithm<float> sAlgo(numClasses);
   sAlgo.initSearchLoop(_pTrainingData);
   
   float halfTheta;
   if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
      halfTheta = _theta/2.0;
   else
      halfTheta = 0;

   int numOfDimensions = _maxNumOfDimensions;
   for (int j = 0; j < numColumns; ++j)
   {
      // Tricky way to select numOfDimensions columns randomly out of numColumns
      int rest = numColumns - j;
      float r = rand()/static_cast<float>(RAND_MAX);

      if ( static_cast<float>(numOfDimensions) / rest > r ) 
      {
         --numOfDimensions;
         const pair<vpIterator,vpIterator> dataBeginEnd = 
			 static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(j);
		 

         const vpIterator dataBegin = dataBeginEnd.first;
         const vpIterator dataEnd = dataBeginEnd.second;

         // also sets mu, tmpV, and bestHalfEdge
         tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                          halfTheta, &mu, &tmpV);

         if (tmpThreshold == tmpThreshold) // tricky way to test Nan
         { 
            // small inconsistency compared to the standard algo (but a good
            // trade-off): in findThreshold we maximize the edge (suboptimal but
            // fast) but here (among dimensions) we minimize the energy.
            tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);

            if (tmpEnergy < bestEnergy && tmpAlpha > 0)
            {
               // Store it in the current weak hypothesis.
               // note: I don't really like having so many temp variables
               // but the alternative would be a structure, which would need
               // to be inheritable to make things more consistent. But this would
               // make it less flexible. Therefore, I am still undecided. This
               // might change!

               _alpha = tmpAlpha;
               _v = tmpV;
               _selectedColumn = j;
               _threshold = tmpThreshold;

               bestEnergy = tmpEnergy;
            }
         } // tmpThreshold == tmpThreshold
      }
   }

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

   
   return bestEnergy;

   
}

// ------------------------------------------------------------------------------

float DirichletSingleStumpLearner::phi(float val, int /*classIdx*/) const
{
   if (val > _threshold)
      return +1;
   else
      return -1;
}

// ------------------------------------------------------------------------------

float DirichletSingleStumpLearner::phi(InputData* pData,int pointIdx) const
{
   return phi(pData->getValue(pointIdx,_selectedColumn),0);
}

// -----------------------------------------------------------------------

void DirichletSingleStumpLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   FeaturewiseLearner::save(outputStream, numTabs);

   // save selectedCoulumn
   outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
   
}

// -----------------------------------------------------------------------

void DirichletSingleStumpLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   FeaturewiseLearner::load(st);

   _threshold = UnSerialization::seekAndParseEnclosedValue<float>(st, "threshold");

   stringstream thresholdString;
   thresholdString << _threshold;
   _id = _id + thresholdString.str();
}

// -----------------------------------------------------------------------

void DirichletSingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
{
   FeaturewiseLearner::subCopyState(pBaseLearner);

   DirichletSingleStumpLearner* pDirichletSingleStumpLearner =
      dynamic_cast<DirichletSingleStumpLearner*>(pBaseLearner);

   pDirichletSingleStumpLearner->_threshold = _threshold;
}

// -----------------------------------------------------------------------

//void DirichletSingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
//{
//   const int numClasses = pData->getNumClasses();
//   const int numExamples = pData->getNumExamples();
//
//   // reason ignored for the moment as it is used for a single task
//   data.resize( numClasses + numExamples );
//
//   int pos = 0;
//
//   for (int l = 0; l < numClasses; ++l)
//      data[pos++] = _v[l];
//
//   for (int i = 0; i < numExamples; ++i)
//      data[pos++] = DirichletSingleStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
//}

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
