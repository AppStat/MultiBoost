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


#include "BanditSingleSparseStump.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"
#include "Algorithms/StumpAlgorithmLSHTC.h"
#include "Algorithms/ConstantAlgorithmLSHTC.h"
#include "WeakLearners/SingleSparseStumpLearner.h"

#include "Bandits/Exp3G2.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace MultiBoost {

    //REGISTER_LEARNER_NAME(SingleStump, BanditSingleSparseStump)


    // ------------------------------------------------------------------------------

    void BanditSingleSparseStump::init() {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
        const int armNumber = _banditAlgo->getArmNumber();

        if ( numColumns < armNumber )
        {
            cerr << "The number of colums smaller than the number of the arms!!!!!!" << endl;
            exit( -1 );
        }

        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner("SingleSparseStumpLearner");

        _banditAlgo->setArmNumber( numColumns );

        vector<AlphaReal> initialValues( numColumns );

        for( int i=0; i < numColumns; i++ )
        {
            SingleSparseStumpLearner* singleStump = dynamic_cast<SingleSparseStumpLearner*>( pWeakHypothesisSource->create());

            singleStump->setTrainingData(_pTrainingData);
            AlphaReal energy = singleStump->run( i );
            AlphaReal edge = singleStump->getEdge();
            AlphaReal reward = getRewardFromEdge( (AlphaReal) edge );

            initialValues[i] = reward;

            delete singleStump;
        }

        _banditAlgo->initialize( initialValues );

    }

    //-------------------------------------------------------------------------------

    AlphaReal BanditSingleSparseStump::run()
    {

        if ( ! this->_banditAlgo->isInitialized() ) {
            init();
        }

        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();

        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( (AlphaReal) 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * (AlphaReal)0.01 );

        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions

        FeatureReal tmpThreshold;
        AlphaReal tmpAlpha;

        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
        AlphaReal tmpEnergy;

        StumpAlgorithmLSHTC<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);

        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/(AlphaReal)2.0;
        else
            halfTheta = 0;
                
        AlphaReal bestReward = 0.0;


        _banditAlgo->getKBestAction( _K, _armsForPulling );
        _rewards.resize( _armsForPulling.size() );

        if ( this->_armsForPulling.size() == 0 )
        {
            cout << "error" << endl;
        }

        for( int i = 0; i < (int)_armsForPulling.size(); i++ ) {
            //columnIndices[i] = p.second;                  


            const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
                static_cast<SortedData*>(_pTrainingData)->getFilteredReverseBeginEnd( _armsForPulling[i] );

            /*
              const pair<vpIterator,vpIterator> dataBeginEnd = 
              static_cast<SortedData*>(_pTrainingData)->getFilteredBeginEnd( _armsForPulling[i] );
            */

            const vpReverseIterator dataBegin = dataBeginEnd.first;
            const vpReverseIterator dataEnd = dataBeginEnd.second;

            /*
              const vpIterator dataBegin = dataBeginEnd.first;
              const vpIterator dataEnd = dataBeginEnd.second;
            */

            // also sets mu, tmpV, and bestHalfEdge
            tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                             halfTheta, &mu, &tmpV);

            tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
            //update the weights in the UCT tree

            AlphaReal edge = 0.0;
            for ( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin );
            AlphaReal reward = this->getRewardFromEdge( edge );
            _rewards[i] = reward;

            if ( _verbose > 3 ) {
                //cout << "\tK = " <<i << endl;
                cout << "\tTempAlpha: " << tmpAlpha << endl;
                cout << "\tTempEnergy: " << tmpEnergy << endl;
                cout << "\tUpdate weight: " << reward << endl;
            }


            if ( (i==0) || (tmpEnergy < bestEnergy && tmpAlpha > 0) )
            {
                // Store it in the current weak hypothesis.
                // note: I don't really like having so many temp variables
                // but the alternative would be a structure, which would need
                // to be inheritable to make things more consistent. But this would
                // make it less flexible. Therefore, I am still undecided. This
                // might change!

                _alpha = tmpAlpha;
                _v = tmpV;
                _selectedColumn = _armsForPulling[i];
                _threshold = tmpThreshold;

                bestEnergy = tmpEnergy;
                bestReward = reward;
            }
        }

        if ( _banditAlgoName == BA_EXP3G2 )
        {
            vector<AlphaReal> ePayoffs( numColumns );                       
            fill( ePayoffs.begin(), ePayoffs.end(), 0.0 );

            for( int i=0; i<_armsForPulling.size(); i++ )
            {
                ePayoffs[_armsForPulling[i]] = _rewards[i];
            }               
            estimatePayoffs( ePayoffs );

            (dynamic_cast<Exp3G2*>(_banditAlgo))->receiveReward( ePayoffs );
        } else {
            for( int i=0; i<_armsForPulling.size(); i++ )
            {
                _banditAlgo->receiveReward( _armsForPulling[i], _rewards[i] );
            }               
        }

        if ( _verbose > 2 ) cout << "Column has been selected: " << _selectedColumn << endl;

        stringstream thresholdString;
        thresholdString << _threshold;
        _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

        _reward = bestReward;

        return bestEnergy;
    }

    // ------------------------------------------------------------------------------

    AlphaReal BanditSingleSparseStump::run( int colIdx )
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();

        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );

        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions

        AlphaReal tmpAlpha;

        AlphaReal bestEnergy = numeric_limits<float>::max();

        StumpAlgorithmLSHTC<FeatureReal> sAlgo(numClasses);
        sAlgo.initSearchLoop(_pTrainingData);

        AlphaReal halfTheta;
        if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
            halfTheta = _theta/2.0;
        else
            halfTheta = 0;


        const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
            static_cast<SortedData*>(_pTrainingData)->getFilteredReverseBeginEnd( colIdx );


        const vpReverseIterator dataBegin = dataBeginEnd.first;
        const vpReverseIterator dataEnd = dataBeginEnd.second;

        // also sets mu, tmpV, and bestHalfEdge
        _threshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
                                                       halfTheta, &mu, &tmpV);

        bestEnergy = getEnergy(mu, tmpAlpha, tmpV);

        _alpha = tmpAlpha;
        _v = tmpV;
        _selectedColumn = colIdx;

        stringstream thresholdString;
        thresholdString << _threshold;
        _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();


        return bestEnergy;

    }

    // -----------------------------------------------------------------------

} // end of namespace MultiBoost
