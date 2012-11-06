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


#include "ProductLearner.h"

#include "IO/Serialization.h"
#include "Others/Example.h"
#include "Utils/StreamTokenizer.h"

#include <math.h>
#include <limits>

namespace MultiBoost {

	//REGISTER_LEARNER_NAME(Product, ProductLearner)
	REGISTER_LEARNER(ProductLearner)

		// -----------------------------------------------------------------------

		void ProductLearner::declareArguments(nor_utils::Args& args)
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("baselearnertype", 
			"The name of the learner that serves as a basis for the product\n"
			"  and the number of base learners to be multiplied\n"
			"  Don't forget to add its parameters\n",
			2, "<baseLearnerType> <numBaseLearners>");

	}

	// ------------------------------------------------------------------------------

	void ProductLearner::initLearningOptions(const nor_utils::Args& args)
	{
		BaseLearner::initLearningOptions(args);

		string baseLearnerName;
		args.getValue("baselearnertype", 0, baseLearnerName);   
		args.getValue("baselearnertype", 1, _numBaseLearners);   

		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
		pWeakHypothesisSource->initLearningOptions(args);

		for( int ib = 0; ib < _numBaseLearners; ++ib ) {
			_baseLearners.push_back(pWeakHypothesisSource->create());
			_baseLearners[ib]->initLearningOptions(args);
		}
	}

	// ------------------------------------------------------------------------------

	AlphaReal ProductLearner::classify(InputData* pData, int idx, int classIdx)
	{
		AlphaReal result  = 1;
		for( int ib = 0; ib < _numBaseLearners; ++ib )
			result *= _baseLearners[ib]->classify( pData, idx, classIdx );
		return result;
	}

	// ------------------------------------------------------------------------------

	AlphaReal ProductLearner::run()
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numExamples = _pTrainingData->getNumExamples();

		// Backup original labels
		for (int i = 0; i < numExamples; ++i) {
			const vector<Label>& labels = _pTrainingData->getLabels(i);
			vector<char> exampleLabels;
			for (int l = 0; l < numClasses; ++l)
				exampleLabels.push_back(labels[l].y);
			_savedLabels.push_back(exampleLabels);
		}

		for(int ib = 0; ib < _numBaseLearners; ++ib)
			_baseLearners[ib]->setTrainingData(_pTrainingData);

		AlphaReal energy = numeric_limits<AlphaReal>::max();
		AlphaReal previousEnergy, hx, previousAlpha;
		BaseLearner* pPreviousBaseLearner = 0;

		bool firstLoop = true;
		int ib = -1;
		while (1) {
			ib += 1;
			if (ib >= _numBaseLearners) {
				ib = 0;
				firstLoop = false;
			}
			previousEnergy = energy;
			previousAlpha = _alpha;
			if (pPreviousBaseLearner)
				delete pPreviousBaseLearner;
			if ( !firstLoop ) {
				// take the old learner off the labels
				for (int i = 0; i < numExamples; ++i) {
					vector<Label>& labels = _pTrainingData->getLabels(i);
					for (int l = 0; l < numClasses; ++l) {
						// Here we could have the option of using confidence rated setting so the
						// real valued output of classify instead of its sign
						hx = _baseLearners[ib]->classify(_pTrainingData,i,l);
						if ( hx < 0 )
							labels[l].y *= -1;
						else if ( hx == 0 ) { // have to redo the multiplications, haven't been tested
							for(int ib1 = 0; ib1 < _numBaseLearners && labels[l].y != 0; ++ib1) {
								if (ib != ib1) {
									hx = _baseLearners[ib1]->classify(_pTrainingData,i,l);
									if (hx < 0)
										labels[l].y *= -1;
									else if (hx == 0)
										labels[l].y = 0;
								}
							}
						}
					}
				}
			}
			pPreviousBaseLearner = _baseLearners[ib]->copyState();
			energy = _baseLearners[ib]->run();
			_alpha = _baseLearners[ib]->getAlpha();
			if (_verbose > 2) {
				cout << "E[" << (ib+1) <<  "] = " << energy << endl << flush;
				cout << "alpha[" << (ib+1) <<  "] = " << _alpha << endl << flush;
			}
			for (int i = 0; i < numExamples; ++i) {
				vector<Label>& labels = _pTrainingData->getLabels(i);
				for (int l = 0; l < numClasses; ++l) {
					// Here we could have the option of using confidence rated setting so the
					// real valued output of classify instead of its sign
					if (labels[l].y != 0) { // perhaps replace it by nor_utils::is_zero(labels[l].y)
						hx = _baseLearners[ib]->classify(_pTrainingData,i,l);
						if ( hx < 0 )
							labels[l].y *= -1;
						else if ( hx == 0 )
							labels[l].y = 0;
					}
				}
			}

			// We have to do at least one full iteration. For real it's not guaranteed
			// Alternatively we could initialize all of them to constant
			//      if ( !firstLoop && energy >= previousEnergy ) {
			//	 if (energy > previousEnergy) {
			//	    _baseLearners[ib] = pPreviousBaseLearner->copyState();
			//           delete pPreviousBaseLearner;
			//	    energy = previousEnergy;
			//	    _alpha = _baseLearners[ib]->getAlpha();
			//	 }
			//	 break;
			//      }
			if ( energy >= previousEnergy ) {
				_alpha = previousAlpha;
				energy = previousEnergy;
				if (firstLoop) {
					for(int ib2 = ib; ib2 < _numBaseLearners; ++ib2)
						delete _baseLearners[ib2];
					_numBaseLearners = ib;
				}
				else {
					_baseLearners[ib] = pPreviousBaseLearner->copyState();
				}
				delete pPreviousBaseLearner;
				break;
			} 
		}

		// Restore original labels
		for (int i = 0; i < numExamples; ++i) {
			vector<Label>& labels = _pTrainingData->getLabels(i);
			for (int l = 0; l < numClasses; ++l)
				labels[l].y = _savedLabels[i][l];
		}

		_id = _baseLearners[0]->getId();
		for(int ib = 1; ib < _numBaseLearners; ++ib)
			_id += "_x_" + _baseLearners[ib]->getId();
		return energy;
	}

	// -----------------------------------------------------------------------

	void ProductLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		BaseLearner::save(outputStream, numTabs);

		// save numBaseLearners
		outputStream << Serialization::standardTag("numBaseLearners", _numBaseLearners, numTabs) << endl;

		for( int ib = 0; ib < _numBaseLearners; ++ib )
			_baseLearners[ib]->save(outputStream, numTabs + 1);
	}

	// -----------------------------------------------------------------------

	void ProductLearner::load(nor_utils::StreamTokenizer& st)
	{
		BaseLearner::load(st);

		_numBaseLearners = UnSerialization::seekAndParseEnclosedValue<int>(st, "numBaseLearners");
		//   _numBaseLearners = 2;

		for(int ib = 0; ib < _numBaseLearners; ++ib)
			UnSerialization::loadHypothesis(st, _baseLearners, _pTrainingData, _verbose);

	}

	// -----------------------------------------------------------------------

	void ProductLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		BaseLearner::subCopyState(pBaseLearner);

		ProductLearner* pProductLearner =
			dynamic_cast<ProductLearner*>(pBaseLearner);

		pProductLearner->_numBaseLearners = _numBaseLearners;

		// deep copy
		for(int ib = 0; ib < _numBaseLearners; ++ib)
			pProductLearner->_baseLearners.push_back(_baseLearners[ib]->copyState());
	}

	// -----------------------------------------------------------------------

} // end of namespace MultiBoost
