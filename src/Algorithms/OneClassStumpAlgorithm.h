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
 * \file OneClassStumpAlgorithm.h The Decision Stump-based algorithms.
 */

#ifndef __ONECLASS_STUMP_ALGORITHM_H
#define __ONECLASS_STUMP_ALGORITHM_H

#include <vector>
#include <cassert>
#include <math.h>
#include <algorithm>

#include "IO/InputData.h"
#include "Others/Rates.h"
#include "IO/NameMap.h"
#include "Algorithms/ConstantAlgorithm.h"
#include "Algorithms/StumpAlgorithm.h"

using namespace std;

namespace MultiBoost {

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    /**
     * Class specialized in solving decision stump-type algorithms.
     * A decision stump is a decision tree with a single level.
     */
    template <typename T>
        class OneClassStumpAlgorithm : public StumpAlgorithm<T>
    {
    public:
        // A couple of useful typedefs
        /**
         * Iterator on Pair. The pair refers to <index, value>.
         */
        typedef typename vector< pair<int, T> >::const_iterator       vpIterator;
        /**
         * Const iterator on Pair. The pair refers to <index, value>.
         */
        typedef typename vector< pair<int, T> >::const_iterator cvpIterator;
                
    OneClassStumpAlgorithm( int numClasses ) : StumpAlgorithm<T>(numClasses) {}
                
                
        FeatureReal findSingleThresholdWithInit(const vpIterator& dataBegin,
                                                const vpIterator& dataEnd,
                                                InputData* pData,
                                                AlphaReal halfTheta,
                                                vector<sRates>* pMu = NULL, vector<AlphaReal>* pV = NULL);
                
                
    };

    //////////////////////////////////////////////////////////////////////////
    template <typename T> 
        FeatureReal OneClassStumpAlgorithm<T>::findSingleThresholdWithInit
        (const vpIterator& dataBegin,const vpIterator& dataEnd,
         InputData* pData, AlphaReal halfTheta, vector<sRates>* pMu, vector<AlphaReal>* pV)
    { 
        const int numClasses = pData->getNumClasses();
                
        vpIterator currentSplitPos; // the iterator of the currently examined example
        vpIterator previousSplitPos; // the iterator of the example before the current example
        vector<vpIterator> bestSplitPos(numClasses); // the iterator of the best split
        vector<vpIterator> bestPreviousSplitPos(numClasses); // the iterator of the example before the best split
                                
        // initialize halfEdges to the constant classifier's half edges 
        copy(this->_constantHalfEdges.begin(), this->_constantHalfEdges.end(), this->_halfEdges.begin());
                
        //vector<float> currHalfEdges(numClasses);
        vector<AlphaReal> bestHalfEdges(numClasses);
        vector<vector<AlphaReal> > bestHalfEdgesForAllClass(numClasses);
                
        //fill(currHalfEdges.begin(),currHalfEdges.end(),0);
        fill(bestHalfEdges.begin(),bestHalfEdges.end(),0);
                                
        for( int l=0; l<numClasses; ++l )
            bestHalfEdgesForAllClass[l].resize(numClasses);
                                
        vector<Label>::const_iterator lIt;
                
        // find the best threshold (cutting point)
        // at the first split we have
        // first split: x | x x x x x x x x ..
        //    previous -^   ^- current
        for( currentSplitPos = previousSplitPos = dataBegin, ++currentSplitPos;
             currentSplitPos != dataEnd; 
             previousSplitPos = currentSplitPos, ++currentSplitPos)
        {
            vector<Label>& labels = pData->getLabels(previousSplitPos->first);
                        
            // recompute halfEdges at the next point
            ////// Bottleneck BEGIN
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                this->_halfEdges[ lIt->idx ] -= lIt->weight * lIt->y;
            ////// Bottleneck END
                        
            // points with the same value of data: to skip because we cannot find a cutting point here!
            // so we only do the cutting if there is a "hole":
            if ( previousSplitPos->second != currentSplitPos->second ) 
            {
                                
                                
                ////// Bottleneck BEGIN
                // the current edge is the new maximum
                for (int l = 0; l < numClasses; ++l)
                {
                    if (fabs(this->_halfEdges[l]) >= fabs(bestHalfEdges[l]))
                    {
                                                
                        bestHalfEdges[l] = this->_halfEdges[l];                                         
                        bestSplitPos[l] = currentSplitPos; 
                        bestPreviousSplitPos[l] = previousSplitPos;
                                                
                        for(int k = 0; k < numClasses; ++k) {
                            bestHalfEdgesForAllClass[l][k] = this->_halfEdges[k]; // kind of redundancy since the diagonal elements are stored twice
                        }
                    }
                }
                ////// Bottleneck END                           
            }
        }
                
        int bestClassInd = -1;
        AlphaReal bestHalfEdge = 0;
        // find best class
        for (int l = 0; l < numClasses; ++l)
        {
            if (fabs(bestHalfEdges[l])>=fabs(bestHalfEdge))
            {
                bestClassInd = l;
                bestHalfEdge = bestHalfEdges[l];
            }
        }
                
        if (!nor_utils::is_zero(bestHalfEdge))
        {
            FeatureReal threshold = static_cast<FeatureReal>( bestPreviousSplitPos[bestClassInd]->second + 
                                                              bestSplitPos[bestClassInd]->second ) / 2;
                        
            // Fill the mus if present. This could have been done in the threshold loop, 
            // but here is done just once
            if ( pMu ) 
            {
                for (int l = 0; l < numClasses; ++l)
                {
                    if ( l != bestClassInd )
                    {
                        // **here
                        if (bestHalfEdge > 0)
                            (*pV)[l] = -1;
                        else
                            (*pV)[l] = +1;
                    } else {
                        // we set v to +1 or -1 only for the best class
                        if (bestHalfEdge > 0)
                            (*pV)[l] = +1;
                        else
                            (*pV)[l] = -1;                                  
                    }
                                        
                    (*pMu)[l].classIdx = l;
                                        
                    (*pMu)[l].rPls  = this->_halfWeightsPerClass[l] + (*pV)[l] * bestHalfEdgesForAllClass[bestClassInd][l];
                    (*pMu)[l].rMin  = this->_halfWeightsPerClass[l] - (*pV)[l] * bestHalfEdgesForAllClass[bestClassInd][l];
                    (*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
                }
            }
            //cout << 2 * bestHalfEdge << endl << flush;
            return threshold;
        } else {
            return numeric_limits<FeatureReal>::signaling_NaN();
        }
    } // end of findSingleThresholdWithInit
        
                
} // end of namespace MultiBoost

#endif // __STUMP_ALGORITHM_H
