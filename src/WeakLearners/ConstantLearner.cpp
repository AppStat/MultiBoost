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


#include "ConstantLearner.h"

#include "IO/Serialization.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>

namespace MultiBoost {


//REGISTER_LEARNER_NAME(Constant, ConstantLearner)


    // ------------------------------------------------------------------------------

    AlphaReal ConstantLearner::run()
    {

        const int numClasses = _pTrainingData->getNumClasses();
        //const int numColumns = _pTrainingData->getNumColumns();

        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );

        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions

        ConstantAlgorithm cAlgo;
        cAlgo.findConstant(_pTrainingData,&mu,&tmpV);
   
        _v = tmpV;

        return getEnergy(mu, _alpha, _v);
    }

// ------------------------------------------------------------------------------

    AlphaReal ConstantLearner::run( int colNum )
    {
        return this->run();
    }

        
// -----------------------------------------------------------------------
    void ConstantLearner::initLearning()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        _v.resize(numClasses);
        fill(_v.begin(), _v.end(), 0.0 );

    }
// -----------------------------------------------------------------------      
    void ConstantLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        StochasticLearner::subCopyState( pBaseLearner );
        AbstainableLearner::subCopyState( pBaseLearner );
    }
        
        
// -----------------------------------------------------------------------      
    AlphaReal ConstantLearner::finishLearning()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        AlphaReal bestEdge = 0.0;
        
        for(int i=0; i<numClasses;++i) 
        {
            bestEdge += (_v[i]<0)? -_v[i] : _v[i];  
            _v[i] = (_v[i]<0)? 1.0 : -1.0;  
        }
        
        return bestEdge;
    }       
// -----------------------------------------------------------------------
    AlphaReal ConstantLearner::update( int idx )
    {
        vector<Label> labels = _pTrainingData->getLabels(idx);
        vector<Label>::iterator lIt;
        
        for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
        {
            _v[lIt->idx] += (lIt->weight * lIt->y);
        }
        
        return 0.0;
    }       
        
// -----------------------------------------------------------------------      
    void ConstantLearner::getStateData( vector<FeatureReal>& data, const string& /*reason*/, InputData* pData )
    {
//    const int numClasses = ClassMappings::getNumClasses();
//    const int numExamples = pData->getNumExamples();

//    // reason ignored for the moment as it is used for a single task
//    data.resize( numClasses + numExamples );

//    int pos = 0;

//    for (int l = 0; l < numClasses; ++l)
//       data[pos++] = _v[l];

//    for (int i = 0; i < numExamples; ++i)
//       data[pos++] = ConstantLearner::phi( pData->getValue( i, _selectedColumn), 0 );
    }

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
