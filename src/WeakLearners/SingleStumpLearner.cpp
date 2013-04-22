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


#include "SingleStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"
#include "Algorithms/StumpAlgorithm.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace MultiBoost {
        
    //REGISTER_LEARNER_NAME(SingleStump, SingleStumpLearner)

        
    // ------------------------------------------------------------------------------
        
    AlphaReal SingleStumpLearner::run()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
                
        FeatureReal tmpThreshold;
        AlphaReal tmpAlpha;
                
        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
        AlphaReal tmpEnergy;
                
        StumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        AlphaReal halfTheta;
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
                //if ( static_cast<SortedData*>(_pTrainingData)->isAttributeEmpty( j ) ) continue;
                
                const pair<pair<vpIterator,vpIterator>,
                           pair<vpReverseIterator,vpReverseIterator> > dataSR = 
                    static_cast<SortedData*>(_pTrainingData)->getFilteredandReverseBeginEnd(j);
                
                const vpIterator dataBegin = dataSR.first.first;
                const vpIterator dataEnd = dataSR.first.second;
                const vpReverseIterator dataReverseBegin = dataSR.second.first;
                const vpReverseIterator dataReverseEnd = dataSR.second.second;
                
                FeatureReal mostFrequentFeatureValue = _pTrainingData->getMostFrequentValuePerFeature()[j];
                
                // also sets mu, tmpV, and bestHalfEdge
                tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, dataReverseBegin, dataReverseEnd,
                                                                 _pTrainingData, halfTheta, &mu, &tmpV,mostFrequentFeatureValue);
                                
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
                
        if ( _selectedColumn != -1 )
        {
            stringstream thresholdString;
            thresholdString << _threshold;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();    
        } else {
            bestEnergy = numeric_limits<AlphaReal>::signaling_NaN();
        }
                
                
        return bestEnergy;
                
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SingleStumpLearner::run( int colIdx )
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
                
        AlphaReal tmpAlpha;
                
        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
                
        StumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/2.0;
        else
            halfTheta = 0;
                
        int numOfDimensions = _maxNumOfDimensions;
                
        const pair<pair<vpIterator,vpIterator>,
                   pair<vpReverseIterator,vpReverseIterator> > dataSR = 
            static_cast<SortedData*>(_pTrainingData)->getFilteredandReverseBeginEnd(colIdx);

        const vpIterator dataBegin = dataSR.first.first;
        const vpIterator dataEnd = dataSR.first.second;
        const vpReverseIterator dataReverseBegin = dataSR.second.first;
        const vpReverseIterator dataReverseEnd = dataSR.second.second;

        FeatureReal mostFrequentFeatureValue = _pTrainingData->getMostFrequentValuePerFeature()[colIdx];
        
        // also sets mu, tmpV, and bestHalfEdge
        _threshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, dataReverseBegin, dataReverseEnd,
                                                       _pTrainingData, halfTheta, &mu, &tmpV, mostFrequentFeatureValue);
        
        bestEnergy = getEnergy(mu, tmpAlpha, tmpV);
                
        _alpha = tmpAlpha;
        _v = tmpV;
        _selectedColumn = colIdx;
                
        if ( _selectedColumn != -1 )
        {
            stringstream thresholdString;
            thresholdString << _threshold;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();    
        } else {
            bestEnergy = numeric_limits<AlphaReal>::signaling_NaN();
        }
                
        return bestEnergy;
                
    }
        
    // ------------------------------------------------------------------------------
    AlphaReal SingleStumpLearner::run( vector<int>& colIndexes )
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
                
        FeatureReal tmpThreshold;
        AlphaReal tmpAlpha;
                
        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
        AlphaReal tmpEnergy;
                
        StumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/2.0;
        else
            halfTheta = 0;
                
        int numOfDimensions = _maxNumOfDimensions;
        for (int j = 0; j < (int)colIndexes.size(); ++j)
        {

            const pair<pair<vpIterator,vpIterator>,
                       pair<vpReverseIterator,vpReverseIterator> > dataSR = 
                static_cast<SortedData*>(_pTrainingData)->getFilteredandReverseBeginEnd(j);

            const vpIterator dataBegin = dataSR.first.first;
            const vpIterator dataEnd = dataSR.first.second;
            const vpReverseIterator dataReverseBegin = dataSR.second.first;
            const vpReverseIterator dataReverseEnd = dataSR.second.second;
            
            FeatureReal mostFrequentFeatureValue = _pTrainingData->getMostFrequentValuePerFeature()[colIndexes[j]];
            
            // also sets mu, tmpV, and bestHalfEdge
            tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, dataReverseBegin, dataReverseEnd,
                                                             _pTrainingData, halfTheta, &mu, &tmpV,mostFrequentFeatureValue);            
            
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
                    _selectedColumn = colIndexes[j];
                    _threshold = tmpThreshold;
                                        
                    bestEnergy = tmpEnergy;
                }
            } // tmpThreshold == tmpThreshold
        }
                
        if ( _selectedColumn != -1 )
        {
            stringstream thresholdString;
            thresholdString << _threshold;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();
        } else {
            bestEnergy = numeric_limits<AlphaReal>::signaling_NaN();
        }
                
        return bestEnergy;
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SingleStumpLearner::phi(FeatureReal val ) const
    {
        if (val > _threshold)
            return +1;
        else
            return -1;
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SingleStumpLearner::phi(InputData* pData,int pointIdx) const
    {
        return phi(pData->getValue(pointIdx,_selectedColumn),0);
    }
        
    // -----------------------------------------------------------------------
        
    void SingleStumpLearner::save(ofstream& outputStream, int numTabs)
    {
        // Calling the super-class method
        FeaturewiseLearner::save(outputStream, numTabs);
                
        // save selectedCoulumn
        outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
                
    }
        
    // -----------------------------------------------------------------------
        
    void SingleStumpLearner::load(nor_utils::StreamTokenizer& st)
    {
        // Calling the super-class method
        FeaturewiseLearner::load(st);
                
        _threshold = UnSerialization::seekAndParseEnclosedValue<FeatureReal>(st, "threshold");
                
        stringstream thresholdString;
        thresholdString << _threshold;
        _id = _id + thresholdString.str();
    }
        
    // -----------------------------------------------------------------------
        
    void SingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        FeaturewiseLearner::subCopyState(pBaseLearner);
                
        SingleStumpLearner* pSingleStumpLearner =
            dynamic_cast<SingleStumpLearner*>(pBaseLearner);
                
        pSingleStumpLearner->_threshold = _threshold;
    }
        
    // -----------------------------------------------------------------------
        
        
        
    //void SingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
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
    //      data[pos++] = SingleStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
    //}
        
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
