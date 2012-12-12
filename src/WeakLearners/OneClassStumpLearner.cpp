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


#include "OneClassStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"
#include "Algorithms/OneClassStumpAlgorithm.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace MultiBoost {
        
    //REGISTER_LEARNER_NAME(SingleStump, SingleStumpLearner)

        
    // ------------------------------------------------------------------------------
        
    AlphaReal OneClassStumpLearner::run()
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
                
        AlphaReal bestEnergy = numeric_limits<float>::max();
        AlphaReal tmpEnergy;
                
        OneClassStumpAlgorithm<FeatureReal> sAlgo(numClasses);
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
                                
                const pair<vpIterator,vpIterator> dataBeginEnd = 
                    static_cast<SortedData*>(_pTrainingData)->getFilteredBeginEnd(j);
                                
                                
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
                
        if ( _selectedColumn > -1 )
        {
            stringstream thresholdString;
            thresholdString << _threshold;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();    
        } else {
            cout << "Warning: no column has selected!!!!" << endl;
            bestEnergy = numeric_limits<float>::signaling_NaN();
        }
                
                
        return bestEnergy;
                
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal OneClassStumpLearner::run( int colIdx )
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
                
        OneClassStumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/2.0;
        else
            halfTheta = 0;
                
        const pair<vpIterator,vpIterator> dataBeginEnd = 
            static_cast<SortedData*>(_pTrainingData)->getFilteredBeginEnd( colIdx );
                
                
        const vpIterator dataBegin = dataBeginEnd.first;
        const vpIterator dataEnd = dataBeginEnd.second;
                
        // also sets mu, tmpV, and bestHalfEdge
        _threshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                       halfTheta, &mu, &tmpV);
                
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
            bestEnergy = numeric_limits<float>::signaling_NaN();
        }
                
        return bestEnergy;
                
    }
        
    // ------------------------------------------------------------------------------
    AlphaReal OneClassStumpLearner::run( vector<int>& colIndexes )
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
                
        OneClassStumpAlgorithm<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);
                
        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/2.0;
        else
            halfTheta = 0;
                
        int numOfDimensions = _maxNumOfDimensions;
        for (int j = 0; j < (int)colIndexes.size(); ++j)
        {
            const pair<vpIterator,vpIterator> dataBeginEnd = 
                static_cast<SortedData*>(_pTrainingData)->getFilteredBeginEnd(colIndexes[j]);
                        
                        
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
        
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
