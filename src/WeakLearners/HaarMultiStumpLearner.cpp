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
 *    Contact: multiboost@googlegroups.com 
 * 
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


#include "HaarMultiStumpLearner.h"

#include "IO/HaarData.h"
#include "IO/Serialization.h"
#include "WeakLearners/Haar/HaarFeatures.h" // for shortname->type and viceversa (see serialization)

#include "Algorithms/StumpAlgorithm.h"

#include <limits> // for numeric_limits
#include <ctime> // for time

namespace MultiBoost {
	
	REGISTER_LEARNER_NAME(HaarMultiStump, HaarMultiStumpLearner)
	
	// ------------------------------------------------------------------------------
	
	void HaarMultiStumpLearner::declareArguments(nor_utils::Args& args)
	{
		// call the superclasses
		HaarLearner::declareArguments(args);
		MultiStumpLearner::declareArguments(args);
	}
	
	// ------------------------------------------------------------------------------
	
	void HaarMultiStumpLearner::initLearningOptions(const nor_utils::Args& args)
	{
		// call the superclasses
		HaarLearner::initOptions(args);
		MultiStumpLearner::initLearningOptions(args);
	}
	
	// ------------------------------------------------------------------------------
	
	AlphaReal HaarMultiStumpLearner::classify(InputData* pData, int idx, int classIdx)
	{
		// The integral image data from the input must be transformed into the 
		// feature's space. This is done by getValue of the selected feature.
		return _v[classIdx] *
		HaarMultiStumpLearner::phi( 
								   _pSelectedFeature->getValue( 
															   pData->getValues(idx), _selectedConfig ),
								   //static_cast<HaarData*>(pData)->getIntImage(idx), _selectedConfig ),
								   classIdx );
		
	}
	
	// ------------------------------------------------------------------------------
	
	AlphaReal HaarMultiStumpLearner::run()
	{
		const int numClasses = _pTrainingData->getNumClasses();
		
		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
		
		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
		
		vector<FeatureReal> tmpThresholds(numClasses);
		AlphaReal tmpAlpha;
		
		AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
		AlphaReal tmpEnergy;
		
		HaarData* pHaarData = static_cast<HaarData*>(_pTrainingData);
		
		// get the whole data matrix
		//   const vector<int*>& intImages = pHaarData->getIntImageVector();
		
		// The data matrix transformed into the feature's space
		vector< pair<int, FeatureReal> > processedHaarData(_pTrainingData->getNumExamples());
		
		// I need to prepare both type of sampling
		int numConf; // for ST_NUM
		time_t startTime, currentTime; // for ST_TIME
		
		long numProcessed;
		bool quitConfiguration;
		
		StumpAlgorithm<FeatureReal> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);
		
		// The declared features types
		vector<HaarFeature*>& loadedFeatures = pHaarData->getLoadedFeatures();
		
		// for every feature type
		vector<HaarFeature*>::iterator ftIt;
		for (ftIt = loadedFeatures.begin(); ftIt != loadedFeatures.end(); ++ftIt)
		{
			// just for readability
			HaarFeature* pCurrFeature = *ftIt;
			if (_samplingType != ST_NO_SAMPLING)
				pCurrFeature->setAccessType(AT_RANDOM_SAMPLING);
			
			// Reset the iterator on the configurations. For random sampling
			// this shuffles the configurations.
			pCurrFeature->resetConfigIterator();
			quitConfiguration = false;
			numProcessed = 0;
			
			numConf = 0;
			time( &startTime );
			
			if (_verbose > 1)
				cout << "Learning type " << pCurrFeature->getName() << ".." << flush;
			
			// While there is a configuration available
			while ( pCurrFeature->hasConfigs() ) 
			{
				// transform the data from intImages to the feature's space
				pCurrFeature->fillHaarData(_pTrainingData->getExamples(), processedHaarData);
				// sort the examples in the new space by their coordinate
				sort( processedHaarData.begin(), processedHaarData.end(), 
					 nor_utils::comparePair<2, int, float, less<float> >() );
				
				// find the optimal threshold
				sAlgo.findMultiThresholdsWithInit(processedHaarData.begin(), processedHaarData.end(), 
												  _pTrainingData, tmpThresholds, &mu, &tmpV);
				
				tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
				++numProcessed;
				
				if (tmpEnergy < bestEnergy)
				{
					// Store it in the current weak hypothesis.
					// note: I don't really like having so many temp variables
					// but the alternative would be a structure, which would need
					// to be inheritable to make things more consistent. But this would
					// make it less flexible. Therefore, I am still undecided. This
					// might change!
					_alpha = tmpAlpha;
					_v = tmpV;
					
					// I need to save the configuration because it changes within the object
					_selectedConfig = pCurrFeature->getCurrentConfig();
					// I save the object because it contains the informations about the type,
					// the name, etc..
					_pSelectedFeature = pCurrFeature;
					_thresholds = tmpThresholds;
					
					bestEnergy = tmpEnergy;
				}
				
				// Move to the next configuration
				pCurrFeature->moveToNextConfig();
				
				// check stopping criterion for random configurations
				switch (_samplingType)
				{
					case ST_NUM:
						++numConf;
						if (numConf >= _samplingVal)
							quitConfiguration = true;
						break;
					case ST_TIME:            
					{
						time( &currentTime );
						float diff = difftime(currentTime, startTime); // difftime is in seconds
						if (diff >= _samplingVal)
							quitConfiguration = true;
					}
						break;
					case ST_NO_SAMPLING:
						perror("ERROR: st no sampling... not sure what this means");
						
						break;
						
				} // end switch
				
				if (quitConfiguration)
					break;
				
			} // end while
			
			if (_verbose > 1)
			{
				time( &currentTime );
				float diff = difftime(currentTime, startTime); // difftime is in seconds
				
				cout << "done! "
				<< "(processed: " << numProcessed
				<< " - elapsed: " << diff << " sec)" 
				<< endl;
			}
			
		}
		
		if (!_pSelectedFeature)
		{
			cerr << "ERROR: No Haar Feature found. Something must be wrong!" << endl;
			exit(1);
		}
		else
		{
			if (_verbose > 1)
				cout << "Selected type: " << _pSelectedFeature->getName() << endl;
		}
		
		return bestEnergy;
	}
	
	// ------------------------------------------------------------------------------
	
	InputData* HaarMultiStumpLearner::createInputData()
	{ 
		return new HaarData();
	}
	
	// ------------------------------------------------------------------------------
	
	AlphaReal HaarMultiStumpLearner::phi(FeatureReal val, int classIdx) const
	{
		if (val > _thresholds[classIdx])
			return +1;
		else
			return -1;
	}
	
	// ------------------------------------------------------------------------------
	
	void HaarMultiStumpLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class methods
		MultiStumpLearner::save(outputStream, numTabs);
		HaarLearner::save(outputStream, numTabs);
	}
	
	// -----------------------------------------------------------------------
	
	void HaarMultiStumpLearner::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class methods
		MultiStumpLearner::load(st);
		HaarLearner::load(st);
	}
	
	// -----------------------------------------------------------------------
	
	void HaarMultiStumpLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		MultiStumpLearner::subCopyState(pBaseLearner);
		HaarLearner::subCopyState(dynamic_cast<HaarLearner*>(pBaseLearner));
	}
	
	// -----------------------------------------------------------------------
	
} // end of MultiBoost namespace
