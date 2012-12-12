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



#include "AdaLineLearner.h"
#include "Utils/Utils.h"

#include <math.h>

using namespace std;

namespace MultiBoost {
        
        

    // ------------------------------------------------------------------------------               
    AlphaReal AdaLineLearner::run()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();                                              
        const int numExamples = _pTrainingData->getNumExamples();
                
        vector<AlphaReal> bestv(numClasses);
                
        _v.resize(numClasses);
                
        AlphaReal gammat = _initialGammat;
                
        if (_verbose>4)
            cout << "-->Init gamma: " << gammat << endl;
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        _featuresWeight.resize(numColumns);
        fill(_featuresWeight.begin(),_featuresWeight.end(), 1.0 );
        nor_utils::normalizeLengthOfVector(_featuresWeight);
                
        _v.resize( numClasses );                
        for(int i=0; i<numClasses; ++i ) 
        {
            _v[i] = (rand() % 2) * 2 - 1;                   
            //fill(vsArray[i].begin(),vsArray[i].end(),0.0);
        }
        nor_utils::normalizeLengthOfVector( _v );
                
                
        if ( _gMethod == OPT_SGD )
        {
            vector<int> randomPermutation(numExamples);
            for (int i = 0; i < numExamples; ++i ) randomPermutation[i]=i;
            random_shuffle( randomPermutation.begin(), randomPermutation.end() );                   
                        
            AlphaReal gammaDivider = 1.0;
            for (int i = 0; i < numExamples; ++i )
            {
                if ((i>0)&&((i%_gammdivperiod)==0)) gammaDivider += 1.0;
                                
                int randomTrainingInstanceIdx = randomPermutation[i];
                vector<Label> labels = _pTrainingData->getLabels(randomTrainingInstanceIdx);
                                
                FeatureReal     innerProduct = 0.0;                             
                for (int j = 0; j < numColumns; ++j)
                {
                    innerProduct += _featuresWeight[j] * _pTrainingData->getValue(randomTrainingInstanceIdx, j);
                }
                                
                vector<AlphaReal> deltaV(numClasses, 0.0);
                switch (_tFunction) {
                case TF_EXPLOSS:
                    for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                    {
                        deltaV[it->idx] = it->weight * exp( - _v[it->idx] * it->y * innerProduct ) *
                            it->y * innerProduct;
                    }                                                       
                    break;
                case TF_EDGE:
                    // has to be implemented
                    break;
                default:
                    break;
                }
                                
                                
                                
                vector<AlphaReal> deltaW(numColumns, 0.0);
                for (int j = 0; j < numColumns; ++j)
                {                                       
                    FeatureReal val = _pTrainingData->getValue(randomTrainingInstanceIdx, j);                                                                                                                       
                                        
                    switch (_tFunction) {
                    case TF_EXPLOSS:
                        for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                        {
                            deltaW[j] += (it->weight * exp( - _v[it->idx] * it->y * innerProduct ) *
                                          val * _v[it->idx] * it->y );
                        }                                                       
                        break;
                    case TF_EDGE:
                        // has to be implemented
                        break;
                    default:
                        break;
                    }
                }
                                
                // gradient step
                for (int j = 0; j < numColumns; ++j)
                {                                       
                    _featuresWeight[j] -= numExamples * static_cast<FeatureReal>(gammat * deltaW[j]);                                                                               
                }
                nor_utils::normalizeLengthOfVector( _featuresWeight );
                                
                for (int j = 0; j < numClasses; ++j)
                {
                    _v[j] -= numExamples * static_cast<FeatureReal>(gammat * deltaV[j]);                                                                            
                }
                nor_utils::normalizeLengthOfVector( _v );
                                
            }
                        
        } else if (_gMethod == OPT_BGD )
        {
        } else {
            cout << "Unknown optimization method!" << endl;
            exit(-1);
        }
                
        for(int k=0; k<numClasses; ++k ) _v[k]= _v[k] < 0 ? -1.0 : 1.0;
                
        //calculate alpha
        this->_alpha = 0.0;
        AlphaReal eps_min = 0.0, eps_pls = 0.0;
                
        //_pTrainingData->clearIndexSet();
        for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) 
        {
            vector< Label> l = _pTrainingData->getLabels( i );
            for( vector< Label >::iterator it = l.begin(); it != l.end(); it++ ) {
                AlphaReal result  = this->classify( _pTrainingData, i, it->idx );
                result *= (it->y * it->weight);
                if ( result < 0 ) eps_min -= result;
                if ( result > 0 ) eps_pls += result;
            }
                        
        }
                
        this->_alpha = getAlpha( eps_min, eps_pls );
                
        if (_verbose>2)
        {
            cout << "---> Alpha: " << this->_alpha << endl;
        }                                                                                 
                
        AlphaReal edge = this->getEdge(true);
                
        return edge;            
    }
        
    // ------------------------------------------------------------------------------                       
    void AdaLineLearner::declareArguments(nor_utils::Args& args)
    {
        AbstainableLearner::initLearningOptions( args );
        StochasticLearner::initLearningOptions( args );
    }
        
    // ------------------------------------------------------------------------------                       
    void AdaLineLearner::initLearningOptions(const nor_utils::Args& args)
    {
        AbstainableLearner::initLearningOptions( args );
        StochasticLearner::initLearningOptions( args );
    }
        
    // ------------------------------------------------------------------------------                       
    void AdaLineLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        AbstainableLearner::subCopyState( pBaseLearner );
        StochasticLearner::subCopyState( pBaseLearner );
    }
        
    // ------------------------------------------------------------------------------               
    void AdaLineLearner::initLearning()
    {
    }
        
    // ------------------------------------------------------------------------------               
    AlphaReal AdaLineLearner::finishLearning()
    {
        AlphaReal retval = 0.0;
        return retval;          
    }
        
    // ------------------------------------------------------------------------------               
        
    AlphaReal AdaLineLearner::update( int idx )
    {
        AlphaReal retval = 0.0;
        return retval;          
    }       
    // ------------------------------------------------------------------------------               
        
} // end of namespace MultiBoost

