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
#include <ctime> // for initilazitaion of random number generator
#include <iostream>

namespace MultiBoost {

	//REGISTER_LEARNER_NAME(SingleStump, DirichletSingleStumpLearner)
	REGISTER_LEARNER(DirichletSingleStumpLearner)

	//-------------------------------------------------------------------------------
	//static class members
	vector< int >		DirichletSingleStumpLearner::_T; // the number of a feature has been selected 
	int					DirichletSingleStumpLearner::_numOfCalling = 0; //number of the single stump learner had been called
	int					DirichletSingleStumpLearner::_regCosntant = 1000;

	//-------------------------------------------------------------------------------

	void DirichletSingleStumpLearner::init() {
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();

		DirichletSingleStumpLearner::_numOfCalling = 0;
		DirichletSingleStumpLearner::_T.resize( _pTrainingData->getNumAttributes() );
		//DirichletSingleStumpLearner::_regCosntant = 1000;		

		srand((unsigned)time(0)); 

	}


	// ------------------------------------------------------------------------------

	float DirichletSingleStumpLearner::run()
	{
	   if ( DirichletSingleStumpLearner::_numOfCalling == 0 ) {
			init();
	   }

		~DirichletSingleStumpLearner::_numOfCalling++; 

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

		//chose an index accroding to the UCB1 policy
		int columnIndex = 0;
		vector<float> featureVals(numColumns);
		float tmpVal;
		float sum = 0.0;
		float randNum = (float) rand() / (float) RAND_MAX;

		for (int j = 0; j < numColumns; ++j) {
			tmpVal = ( (float) DirichletSingleStumpLearner::_T[j] + ( DirichletSingleStumpLearner::_regCosntant / numColumns ) ) 
				/ ( (float) ( DirichletSingleStumpLearner::_numOfCalling + DirichletSingleStumpLearner::_regCosntant ) );
			sum += tmpVal;

			if ( randNum < sum ) {
				columnIndex = j;
				break; 
			}
		}		

		const pair<vpIterator,vpIterator> dataBeginEnd = 
			static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(columnIndex);


		const vpIterator dataBegin = dataBeginEnd.first;
		const vpIterator dataEnd = dataBeginEnd.second;

		// also sets mu, tmpV, and bestHalfEdge
		tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
			halfTheta, &mu, &tmpV);

		// small inconsistency compared to the standard algo (but a good
		// trade-off): in findThreshold we maximize the edge (suboptimal but
		// fast) but here (among dimensions) we minimize the energy.
		bestEnergy = getEnergy(mu, tmpAlpha, tmpV);


		_alpha = tmpAlpha;
		_v = tmpV;
		_selectedColumn = columnIndex;
		_threshold = tmpThreshold;


		DirichletSingleStumpLearner::_T[columnIndex]++;

		cout << "Column to be selected: " << columnIndex << endl;

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
