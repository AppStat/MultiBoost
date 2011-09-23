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
 * \file StumpAlgorithmLSHTC.h The Decision Stump-based algorithms.
 */

#ifndef __STUMP_ALGORITHM_LSHTC_H
#define __STUMP_ALGORITHM_LSHTC_H

#include <vector>
#include <cassert>

#include "IO/InputData.h"
#include "Others/Rates.h"
#include "IO/NameMap.h"
#include "Algorithms/ConstantAlgorithmLSHTC.h"

using namespace std;

namespace MultiBoost {

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    /**
     * Class specialized in solving decision stump-type algorithms.
     * A decision stump is a decision tree with a single level.
     */
    template <typename T>
        class StumpAlgorithmLSHTC
    {
    public:

        // A couple of useful typedefs
        /**
         * Iterator on Pair. The pair refers to <index, value>.
         */
        typedef typename vector< pair<int, T> >::reverse_iterator       vpReverseIterator;
        /**
         * Const iterator on Pair. The pair refers to <index, value>.
         */
        typedef typename vector< pair<int, T> >::const_reverse_iterator cvpReverseIterator;

        StumpAlgorithmLSHTC( int numClasses )
        {
            // resize: it's done here to avoid a reallocation
            // for each dimension.

            _halfEdges.resize(numClasses);
            _constantHalfEdges.resize(numClasses);
            _bestHalfEdges.resize(numClasses);
            //_bestHalfEdgesNegative.resize(numClasses);
            _halfWeightsPerClass.resize(numClasses);   
            //_edgeOfZeroElements.resize(numClasses);
        }

        /**
         * Initilizes halfWeightsPerClass and constantHalfEdges for subsequent calls
         * to findSingleThresholdWithInit or findMultiThresholdWithInit
         * \param pData The pointer to the data class.
         * \date 03/07/2006
         */
        void initSearchLoop(InputData* pData);

        /**
         * Find the optimal threshold that maximizes
         * the edge (or minimizes the error) on the given weighted data.
         * In case of multi-class data, the threshold will minimize the
         * error over all the classes:
         * \param dataBegin The iterator to the beginning of the data.
         * \param dataEnd The iterator to the end of the data.
         * \param pData The pointer to the original data class. Used to obtain the label of
         * the example from its index.
         * \param pMu The The class-wise rates to update. (if provided)
         * \param pV The alignment vector to update. (if provided)
         * \see sRates
         * \return The threshold found.
         * \remark The algorithm (suboptimally) maximizes the Edge. This 
         * is slightly different from the standard AdaBoost which maximizes
         * the energy even in the decision stump algorithm. This is the result
         * of a trade-off between speed and full adherence to the theory.
         * \date 11/11/2005
         */
        FeatureReal findSingleThreshold(const vpReverseIterator& dataBegin,
                                        const vpReverseIterator& dataEnd,
                                        InputData* pData,
                                        AlphaReal halfTheta = 0.0,
                                        vector<sRates>* pMu = NULL, vector<AlphaReal>* pV = NULL);

        /**
         * Same as findSingleThreshold, but the caller has to call initSearchLoop
         * explicitly before the loop over the dimensions (features). It is more efficient 
         * than findSingleThreshold since findSingleThreshold calls initSearchLoop 
         * in each iteration. However, it can only be used if
         * the data points and their weights do not change between initSearchLoop and subsequent
         * calls to findSingleThreshold. See SingleStumpLearner::run for an example.
         * \param dataBegin The iterator to the beginning of the data.
         * \param dataEnd The iterator to the end of the data.
         * \param pData The pointer to the original data class. Used to obtain the label of
         * the example from its index.
         * \param pMu The The class-wise rates to update. (if provided)
         * \param pV The alignment vector to update. (if provided)
         * \see sRates
         * \see findSingleThreshold
         * \return The threshold found.
         * \date 03/07/2006
         */
        FeatureReal findSingleThresholdWithInit(const vpReverseIterator& dataBegin,
                                                const vpReverseIterator& dataEnd,
                                                InputData* pData,
                                                AlphaReal halfTheta,
                                                vector<sRates>* pMu = NULL, vector<AlphaReal>* pV = NULL);

        /**
         * Find the optimal thresholds (one for each class) that maximizes
         * the edge (or minimizes the error) on the given data weighted data.
         * \param dataBegin The iterator to the beginning of the data.
         * \param dataEnd The iterator to the end of the data.
         * \param pData The pointer to the original data class. Used to obtain the label of
         * the example from its index.
         * \param thresholds The thresholds to update.
         * \param pMu The The class-wise rates to update (if provided).
         * \param pV The alignment vector to update (if provided).
         * \remark The algorithm (suboptimally) maximizes the Edge. This 
         * is slightly different from the standard AdaBoost which maximizes
         * the energy even in the decision stump algorithm. This is the result
         * of a trade-off between speed and full adherence to the theory.
         * \see sRates
         * \date 11/11/2005
         */
        void findMultiThresholds(const vpReverseIterator& dataBegin,
                                 const vpReverseIterator& dataEnd,
                                 InputData* pData, vector<FeatureReal>& thresholds,
                                 vector<sRates>* pMu = NULL, vector<AlphaReal>* pV = NULL);

        /**
         * Same as findMultiThreshold, but the caller has to call initSearchLoop
         * explicitly before the loop over the dimensions (features). It is more efficient 
         * than findMultiThreshold since findMultiThreshold calls initSearchLoop 
         * in each iteration. However, it can only be used if
         * the data points and their weights do not change between initSearchLoop and subsequent
         * calls to findMultiThreshold. See MultiStumpLearner::run for an example.
         * \param dataBegin The iterator to the beginning of the data.
         * \param dataEnd The iterator to the end of the data.
         * \param pData The pointer to the original data class. Used to obtain the label of
         * the example from its index.
         * \param thresholds The thresholds to update.
         * \param pMu The The class-wise rates to update. (if provided)
         * \param pV The alignment vector to update. (if provided)
         * \see sRates
         * \see findSingleThreshold
         * \return The threshold found.
         * \date 05/07/2006
         */
        void findMultiThresholdsWithInit(const vpReverseIterator& dataBegin,
                                         const vpReverseIterator& dataEnd,
                                         InputData* pData, 
                                         vector<FeatureReal>& thresholds,
                                         vector<sRates>* pMu = NULL, vector<AlphaReal>* pV = NULL);

    private:

        vector<AlphaReal> _halfEdges; //!< half of the class-wise edges
        vector<AlphaReal> _constantHalfEdges; //!< half of the class-wise edges of the constant classifier
        vector<AlphaReal> _bestHalfEdges; //!< half of the edges of the best found threshold.
        vector<AlphaReal> _halfWeightsPerClass; //!< The half of the total weights per class.
        //vector<float> _edgeOfZeroElements; // The edges of the zero value
        vector<vpReverseIterator> _bestSplitPoss; // the iterator of the best split
        vector<vpReverseIterator> _bestPreviousSplitPoss; // the iterator of the example before the best split

    };

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    template <typename T> 
        void StumpAlgorithmLSHTC<T>::initSearchLoop(InputData* pData)
    {
        ConstantAlgorithmLSHTC cAlgo;
        cAlgo.findConstantWeightsEdges(pData,_halfWeightsPerClass,_constantHalfEdges);
    } // end of initSearchLoop

    //////////////////////////////////////////////////////////////////////////

    template <typename T> 
        FeatureReal StumpAlgorithmLSHTC<T>::findSingleThreshold(const vpReverseIterator& dataBegin,
                                                                const vpReverseIterator& dataEnd,
                                                                InputData* pData,
                                                                AlphaReal halfTheta,
                                                                vector<sRates>* pMu, vector<AlphaReal>* pV)
    { 
        // initialize halfEdges to the constant classifier's half edges 
        initSearchLoop(pData);

        // findSingleThreshold after initialization
        return findSingleThresholdWithInit(dataBegin,dataEnd,pData,pMu,pV,halfTheta);

    } // end of findSingleThreshold

    //////////////////////////////////////////////////////////////////////////


    template <typename T> 
        FeatureReal StumpAlgorithmLSHTC<T>::findSingleThresholdWithInit
        (const vpReverseIterator& dataBegin,const vpReverseIterator& dataEnd,
         InputData* pData, AlphaReal halfTheta, vector<sRates>* pMu, vector<AlphaReal>* pV)
    { 
        const int numClasses = pData->getNumClasses();

        if ( static_cast<SortedData*>(pData)->isFilteredAttributeEmpty() ) { // empty the current feature, so we can use only the constant learner              
            FeatureReal threshold = -numeric_limits<FeatureReal>::max(); // we assume that the missing values are equal to zero

            // Fill the mus if present. This could have been done in the threshold loop, 
            // but here is done just once
            if ( pMu ) 
            {
                for (int l = 0; l < numClasses; ++l)
                {
                    (*pV)[l] = +1;
                    (*pMu)[l].classIdx = l;

                    (*pMu)[l].rPls  = _constantHalfEdges[l];
                    (*pMu)[l].rMin  = _constantHalfEdges[l];
                    (*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
                }

            }                       
            return threshold;                       
        }


        vpReverseIterator currentSplitPos; // the iterator of the currently examined example
        vpReverseIterator previousSplitPos; // the iterator of the example before the current example
        vpReverseIterator bestSplitPos; // the iterator of the best split
        vpReverseIterator bestPreviousSplitPos; // the iterator of the example before the best split

        // initialize halfEdges to the constant classifier's half edges 
        //copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _halfEdges.begin());
        for( int i=0 ; i < _constantHalfEdges.size(); i++ ) _halfEdges[i] = -_constantHalfEdges[i]; // minus constant edges because the reverse iteration

        AlphaReal currHalfEdge = 0;
        AlphaReal bestHalfEdge = -numeric_limits<AlphaReal>::max();
        vector<Label>::const_iterator lIt;
                
        int currentDataIndex = 0;
        FeatureReal currentDataValue = 0.0;
                
        int i = 0;
        // find the best threshold (cutting point)
        // at the first split we have
        // first split: x | x x x x x x x x ..
        //    previous -^   ^- current
        for( i = 0, currentSplitPos = previousSplitPos = dataBegin, ++currentSplitPos;
             currentSplitPos != dataEnd; 
             previousSplitPos = currentSplitPos, currentDataIndex = currentSplitPos->first, currentDataValue = currentSplitPos->second, ++currentSplitPos, i++ )
        {
            //cout << previousSplitPos->second << " " << currentSplitPos->second << endl;
            vector<Label>& labels = pData->getLabels(previousSplitPos->first);

            // recompute halfEdges at the next point
            ////// Bottleneck BEGIN
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                _halfEdges[ lIt->idx ] += lIt->weight * lIt->y;
            ////// Bottleneck END

            // same value of data: to skip because we cannot find a cutting point here!
            if ( previousSplitPos->second != currentSplitPos->second ) 
            {

                currHalfEdge = 0;

                ////// Bottleneck BEGIN
                if ( nor_utils::is_zero(halfTheta) ) { // we save an "if" in the loop, 20% faster
                    for (int l = 0; l < numClasses; ++l) { 
                        // flip the class-wise edge if it is negative
                        // but store the flipping bit only at the end (below**)
                        if ( _halfEdges[l] < 0 )
                            currHalfEdge -= _halfEdges[l];
                        else
                            currHalfEdge += _halfEdges[l];
                    }
                }
                else {
                    for (int l = 0; l < numClasses; ++l) { 
                        // flip the class-wise edge if it is negative
                        // but store the flipping bit only at the end (below**)
                        if ( _halfEdges[l] < halfTheta )
                            currHalfEdge += _halfEdges[l];
                        else if ( _halfEdges[l] < -halfTheta )
                            currHalfEdge -= _halfEdges[l];
                    }
                }
                ////// Bottleneck END

                // the current edge is the new maximum
                if (currHalfEdge > bestHalfEdge)
                {
                    bestHalfEdge = currHalfEdge;
                    bestSplitPos = currentSplitPos; 
                    bestPreviousSplitPos = previousSplitPos; 

                    for (int l = 0; l < numClasses; ++l)
                        _bestHalfEdges[l] = _halfEdges[l];
                                        
                    FeatureReal threshold = ( previousSplitPos->second + currentSplitPos->second ) / 2;
                    //cout << "Current threshold: " << threshold;
                    //cout << endl;
                }
            }
        }

        // we need to store the split position in float because the iterator don't acces the non-zero elements
                
        FeatureReal bestPreviousSplitPosFloat;
        FeatureReal bestSplitPosFloat; 
                
        if (bestHalfEdge > -numeric_limits<AlphaReal>::max()) { //there is only one non-zero value
            bestPreviousSplitPosFloat = bestPreviousSplitPos->second;
            bestSplitPosFloat = bestSplitPos->second; 
        } else {
            //currentDataValue = dataBegin->second;
            //currentDataIndex = dataBegin->first;
            //bestPreviousSplitPosFloat = 0.0;
            //bestSplitPosFloat = 1.0; 
        }


        // dataEnd will contain the smallest value wich isn't equal to zero
        // the edge of this element will be extracted
        vector<Label>& labelsLast = pData->getLabels( currentDataIndex );

        // recompute halfEdges at the next point
        ////// Bottleneck BEGIN
        for (lIt = labelsLast.begin(); lIt != labelsLast.end(); ++lIt )
            _halfEdges[ lIt->idx ] += lIt->weight * lIt->y;
        ////// Bottleneck END

        currHalfEdge = 0;

        ////// Bottleneck BEGIN
        if ( nor_utils::is_zero(halfTheta) ) { // we save an "if" in the loop, 20% faster
            for (int l = 0; l < numClasses; ++l) { 
                // flip the class-wise edge if it is negative
                // but store the flipping bit only at the end (below**)
                if ( _halfEdges[l] > 0 )
                    currHalfEdge -= _halfEdges[l];
                else
                    currHalfEdge += _halfEdges[l];
            }
        }
        else {
            for (int l = 0; l < numClasses; ++l) { 
                // flip the class-wise edge if it is negative
                // but store the flipping bit only at the end (below**)
                if ( _halfEdges[l] < halfTheta )
                    currHalfEdge += _halfEdges[l];
                else if ( _halfEdges[l] < -halfTheta )
                    currHalfEdge -= _halfEdges[l];
            }
        }
        ////// Bottleneck END

        // the current edge is the new maximum
        if (currHalfEdge > bestHalfEdge)
        {
            bestHalfEdge = currHalfEdge;
            bestPreviousSplitPosFloat = currentDataValue;
            bestSplitPosFloat = 0.0; 
                        
            for (int l = 0; l < numClasses; ++l)
                _bestHalfEdges[l] = _halfEdges[l];
        }
        //end of the investigation of the last non-zero elements                

        // If we found a valid stump in this dimension
        if (bestHalfEdge > -numeric_limits<AlphaReal>::max()) 
        {
            FeatureReal threshold = ( bestPreviousSplitPosFloat + bestSplitPosFloat ) / 2;

            // Fill the mus if present. This could have been done in the threshold loop, 
            // but here is done just once
            if ( pMu ) 
            {
                for (int l = 0; l < numClasses; ++l)
                {
                    // **here
                    if (_bestHalfEdges[l] > 0)
                        (*pV)[l] = +1;
                    else
                        (*pV)[l] = -1;

                    (*pMu)[l].classIdx = l;

                    (*pMu)[l].rPls  = _halfWeightsPerClass[l] + (*pV)[l] * _bestHalfEdges[l];
                    (*pMu)[l].rMin  = _halfWeightsPerClass[l] - (*pV)[l] * _bestHalfEdges[l];
                    (*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
                }
            }
            //cout << 2 * bestHalfEdge << endl << flush;
            return threshold;
        }
        else
            return numeric_limits<FeatureReal>::signaling_NaN();

    } // end of findSingleThresholdWithInit

    //////////////////////////////////////////////////////////////////////////

    template <typename T> 
        void StumpAlgorithmLSHTC<T>::findMultiThresholds(const vpReverseIterator& dataBegin,
                                                         const vpReverseIterator& dataEnd,
                                                         InputData* pData, vector<FeatureReal>& thresholds,
                                                         vector<sRates>* pMu, vector<AlphaReal>* pV)
    { 
        // initialize halfEdges to the constant classifier's half edges 
        initSearchLoop(pData);

        // findMultiThreshold after initialization
        return findMultiThresholdsWithInit(dataBegin,dataEnd,pData,thresholds,pMu,pV);

    } // end of findMultiThresholds

    //////////////////////////////////////////////////////////////////////////

    template <typename T> 
        void StumpAlgorithmLSHTC<T>::findMultiThresholdsWithInit
        (const vpReverseIterator& dataBegin, const vpReverseIterator& dataEnd,
         InputData* pData, vector<FeatureReal>& thresholds, 
         vector<sRates>* pMu, vector<AlphaReal>* pV)
    { 
        const int numClasses = pData->getNumClasses();

        vpIterator currentSplitPos; // the iterator of the currently examined example
        vpIterator previousSplitPos; // the iterator of the example before the current example

        // Initializing halfEdges to the constant classifier's half edges 
        copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _halfEdges.begin());

        // Initializing bestHalfEdges, thresholds, and pV to the constant classifier 
        copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _bestHalfEdges.begin());
        for (int l = 0; l < numClasses; ++l)
            thresholds[l] = -numeric_limits<FeatureReal>::max(); // constant cut

        bool alignAlloc = false;
        if (pV == NULL)
        {
            pV = new vector<AlphaReal>(numClasses);
            alignAlloc = true;
        }

        for (int l = 0; l < numClasses; ++l)
        {
            if (_halfEdges[l] > 0) // constant cut
                (*pV)[l] = 1;
            else
                (*pV)[l] = -1;
        }


        vector<Label>::const_iterator lIt;

        // find the best threshold (cutting point)
        // at the first split we have
        // first split: x | x x x x x x x x ..
        //    previous -^   ^- current
        for( currentSplitPos = previousSplitPos = dataBegin, ++currentSplitPos;
             currentSplitPos != dataEnd; 
             previousSplitPos = currentSplitPos, ++currentSplitPos)
        {
            // recompute halfEdges at the next point
            // this is the bottleneck
            vector<Label>& labels = pData->getLabels(previousSplitPos->first);

            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                _halfEdges[ lIt->idx ] -= lIt->weight * lIt->y;

            // points with the same value of data: to skip because we cannot find a cutting point here!
            // so we only do the cutting if there is a "hole":
            if ( previousSplitPos->second != currentSplitPos->second ) 
            {
                for (int l = 0; l < numClasses; ++l)
                { 
                    if (_halfEdges[l] > 0)
                    {
                        // the current edge is the new maximum
                        if (_halfEdges[l] > _bestHalfEdges[l] * (*pV)[l]) 
                        {
                            (*pV)[l] = 1;
                            _bestHalfEdges[l] = _halfEdges[l];
                            thresholds[l] = static_cast<FeatureReal>( previousSplitPos->second +
                                                                      currentSplitPos->second ) / 2;
                        }
                    }
                    else
                    {
                        // the current edge is the new maximum
                        if (-_halfEdges[l] > _bestHalfEdges[l] * (*pV)[l]) 
                        {
                            (*pV)[l] = -1;
                            _bestHalfEdges[l] = _halfEdges[l];
                            thresholds[l] = static_cast<FeatureReal>( previousSplitPos->second +
                                                                      currentSplitPos->second ) / 2;
                        }
                    }
                }
            }
        }

        // Fill the mus if present. This could have been done in the threshold loop, 
        // but here is done just once
        if ( pMu ) 
        {
            for (int l = 0; l < numClasses; ++l)
            {           
                (*pMu)[l].classIdx = l;

                (*pMu)[l].rPls  = _halfWeightsPerClass[l] + (*pV)[l] * _bestHalfEdges[l];
                (*pMu)[l].rMin  = _halfWeightsPerClass[l] - (*pV)[l] * _bestHalfEdges[l];
                (*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
            }
        }

        if (alignAlloc)
            delete pV;

    } // end of findMultiThresholdWithInit

    //////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __STUMP_ALGORITHM_H
