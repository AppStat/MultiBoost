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

#include "MultiStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"

#include "Algorithms/StumpAlgorithm.h"

#include <limits> // for numeric_limits<>
namespace MultiBoost {
        

    // ------------------------------------------------------------------------------
        
    void MultiStumpLearner::declareArguments(nor_utils::Args& args)
    {
        AbstainableLearner::declareArguments(args);
                
        args.declareArgument("rsample",
                             "Instead of searching for a featurewise in all the possible dimensions (features), select a set of "
                             " size <num> of random dimensions. "
                             "Example: -rsample 50 -> Search over only 50 dimensions"
                             "(Turned off for Haar: use -csample instead)",
                             1, "<num>");
                
    }
        
        
    // ------------------------------------------------------------------------------
        
    void MultiStumpLearner::initLearningOptions(const nor_utils::Args& args)
    {
        AbstainableLearner::initLearningOptions(args);
        _maxNumOfDimensions = numeric_limits<int>::max();
                
        // If the sampling is required
        if ( args.hasArgument("rsample") )
            _maxNumOfDimensions = args.getValue<int>("rsample", 0);
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal MultiStumpLearner::run() {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
                
        // set the size of selected columns
        _selectedColumnArray.resize( numClasses );
        fill( _selectedColumnArray.begin(), _selectedColumnArray.end(), -1 );
                
        // for storing the class-wise maximal edge
        vector<AlphaReal> classwiseEdge(numClasses);
        fill(classwiseEdge.begin(), classwiseEdge.end(), -1.0 );
                
        //
        _v.resize(numClasses);
        _thresholds.resize(numClasses);
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal(1.0 / (AlphaReal) _pTrainingData->getNumExamples() * 0.01);
                
        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<sRates> bestmu(numClasses);
                
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
        vector<FeatureReal> tmpThresholds(numClasses);
                
        AlphaReal bestEnergy = numeric_limits<float>::max();

                
        StumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        int numOfDimensions = _maxNumOfDimensions;
        for (int j = 0; j < numColumns; ++j) {
            // Tricky way to select numOfDimensions columns randomly out of numColumns
            int rest = numColumns - j;
            float r = rand() / static_cast<float> (RAND_MAX);
                        
            if (static_cast<float> (numOfDimensions) / rest > r) {
                --numOfDimensions;
                const pair<vpIterator, vpIterator>
                    dataBeginEnd =
                    static_cast<SortedData*> (_pTrainingData)->getFilteredBeginEnd(j);
                                
                const vpIterator dataBegin = dataBeginEnd.first;
                const vpIterator dataEnd = dataBeginEnd.second;
                                
                sAlgo.findMultiThresholdsWithInit(dataBegin, dataEnd,
                                                  _pTrainingData, tmpThresholds, &mu, &tmpV);
                                
                for ( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); ++itR )
                {
                    AlphaReal tmpEdgePerClass = ( itR->rPls - itR->rMin );
                                        
                    // in each iteration the edge will be maximized, here this is done class-wisely.
                    if ( classwiseEdge[itR->classIdx] < tmpEdgePerClass )
                    {
                        classwiseEdge[itR->classIdx] = tmpEdgePerClass;
                        bestmu[itR->classIdx] = *itR;
                                                
                        _v[itR->classIdx] = tmpV[itR->classIdx];
                        _selectedColumnArray[itR->classIdx] = j;
                        _thresholds[itR->classIdx] = tmpThresholds[itR->classIdx];
                        //cout << tmpThresholds[itR->classIdx] << endl;
                    }
                                        
                }
            }
        }
                
        bestEnergy = getEnergy(bestmu, _alpha, _v);
                
        return bestEnergy;
                
    }
    // -----------------------------------------------------------------------
        
    AlphaReal MultiStumpLearner::phi(InputData* pData, int idx, int classIdx) const
    {
        return phi( pData->getValue(idx, _selectedColumnArray[classIdx]), classIdx );
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal MultiStumpLearner::phi(FeatureReal val, int classIdx) const {
        if (val > _thresholds[classIdx])
            return +1;
        else
            return -1;
    }
        
    // -----------------------------------------------------------------------
        
    void MultiStumpLearner::save(ofstream& outputStream, int numTabs) {
        // Calling the super-class method
        AbstainableLearner::save(outputStream, numTabs);
                
        // save all the column indices
        outputStream << Serialization::vectorTag("colArray", _selectedColumnArray,
                                                 _pTrainingData->getClassMap(), "class", (int) 0, numTabs)
                     << endl;
                
        // save all the thresholds
        outputStream << Serialization::vectorTag("thArray", _thresholds,
                                                 _pTrainingData->getClassMap(), "class", (FeatureReal) 0.0, numTabs)
                     << endl;
    }
        
    // -----------------------------------------------------------------------
        
    void MultiStumpLearner::load(nor_utils::StreamTokenizer& st) {
        // Calling the super-class method
        AbstainableLearner::load(st);
                
        // load array of selected column data
        UnSerialization::seekAndParseVectorTag(st, "colArray",
                                               _pTrainingData->getClassMap(), "class", _selectedColumnArray);
                
        // load vArray data
        UnSerialization::seekAndParseVectorTag(st, "thArray",
                                               _pTrainingData->getClassMap(), "class", _thresholds);
    }
        
    // -----------------------------------------------------------------------
        
    void MultiStumpLearner::subCopyState(BaseLearner *pBaseLearner) {
        AbstainableLearner::subCopyState(pBaseLearner);
                
        MultiStumpLearner* pMultiStumpLearner =
            dynamic_cast<MultiStumpLearner*> (pBaseLearner);
                
        pMultiStumpLearner->_thresholds = _thresholds;
        pMultiStumpLearner->_selectedColumnArray = _selectedColumnArray;
    }
        
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
