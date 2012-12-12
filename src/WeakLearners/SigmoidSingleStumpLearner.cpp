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


#include "SigmoidSingleStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"
#include "Algorithms/StumpAlgorithm.h"
#include "Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace MultiBoost {
        
    //REGISTER_LEARNER_NAME(SingleStump, SingleStumpLearner)

    // ------------------------------------------------------------------------------
        
    void SigmoidSingleStumpLearner::declareArguments(nor_utils::Args& args)
    {
        FeaturewiseLearner::declareArguments(args);
        StochasticLearner::declareArguments(args);
    }       
        
    // ------------------------------------------------------------------------------
        
    void SigmoidSingleStumpLearner::initLearningOptions(const nor_utils::Args& args)
    {
        FeaturewiseLearner::initLearningOptions(args);                          
        StochasticLearner::initLearningOptions(args);
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SigmoidSingleStumpLearner::run()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();                                              
        const int numExamples = _pTrainingData->getNumExamples();
                
        FeatureReal bestEdge = -numeric_limits<FeatureReal>::max();
        vector<AlphaReal> bestv(numClasses);
                
        _v.resize(numClasses);
                
        AlphaReal gammat = _initialGammat;
                
        if (_verbose>4)
            cout << "-->Init gamma: " << gammat << endl;
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        vector<FeatureReal> sigmoidSlopes( numColumns );
        vector<FeatureReal> sigmoidOffSets( numColumns );               
                
        vector<vector<AlphaReal> >   vsArray(numColumns);
                
        fill( sigmoidSlopes.begin(), sigmoidSlopes.end(), 0.1 );
        fill( sigmoidOffSets.begin(), sigmoidOffSets.end(), 0.0 );
                
        for(int i=0; i<numColumns; ++i ) 
        {
            vsArray[i].resize(numClasses);
            for (int k=0; k<numClasses; ++k )
            {
                vsArray[i][k] = (rand() % 2) * 2 - 1;
            }
            normalizeLength( vsArray[i] );
            //fill(vsArray[i].begin(),vsArray[i].end(),0.0);
        }
                
                
        if ( _gMethod == OPT_SGD )
        {
            vector<int> randomPermutation(numExamples);
            for (int i = 0; i < numExamples; ++i ) randomPermutation[i]=i;
            random_shuffle( randomPermutation.begin(), randomPermutation.end() );                   
                        
            AlphaReal gammaDivider = 1.0;
            for (int i = 0; i < numExamples; ++i )
            {
                if ((i>0)&&((i%_gammdivperiod)==0)) gammaDivider += 1.0;
                AlphaReal stepOffSet, stepSlope, stepV;
                //int randomTrainingInstanceIdx = (rand() % _pTrainingData->getNumExamples());
                int randomTrainingInstanceIdx = randomPermutation[i];
                vector<Label> labels = _pTrainingData->getLabels(randomTrainingInstanceIdx);
                                
                for (int j = 0; j < numColumns; ++j)
                {
                                        
                    FeatureReal val = _pTrainingData->getValue(randomTrainingInstanceIdx, j);                                       
                    FeatureReal tmpSigVal = sigmoid(val,sigmoidSlopes[j],sigmoidOffSets[j]);
                                        
                                        
                    AlphaReal deltaQ = 0.0;
                                        
                    switch (_tFunction) {
                    case TF_EXPLOSS:
                        for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                        {
                            deltaQ += exp( -vsArray[j][it->idx] * it->y * ( 2*tmpSigVal-1 ) ) 
                                *2.0 * it->weight*vsArray[j][it->idx]*it->y*tmpSigVal*(1.0-tmpSigVal);
                        }
                                                        
                        stepOffSet = -deltaQ;
                        stepSlope = -deltaQ * val;                                                      
                        break;
                    case TF_EDGE:
                        for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                        {
                            deltaQ += 2.0 * it->weight*vsArray[j][it->idx]*it->y*tmpSigVal*(1.0-tmpSigVal);
                        }
                        // because edge should be maximized
                        stepOffSet = -deltaQ;
                        stepSlope = -deltaQ * val;
                        break;
                    default:
                        break;
                    }
                                        
                    // gradient step
                    FeatureReal tmpSigmoidOffSet = sigmoidOffSets[j] - numExamples * static_cast<FeatureReal>(gammat * stepOffSet);                                                                         
                    FeatureReal tmpSigmoidSlopes = sigmoidSlopes[j] - numExamples * static_cast<FeatureReal>(gammat * stepSlope);                                                                   
                                        
                                        
                    // update the parameters
                    for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                    {
                        switch (_tFunction) {
                        case TF_EXPLOSS:
                            stepV = -exp( - vsArray[j][it->idx] * it->y * ( 2*tmpSigVal-1 ) )   
                                * ( it->weight * (2.0 * tmpSigVal - 1.0) * it->y);                                              
                            break;
                        case TF_EDGE:
                            // + gradient since it a maximization task
                            stepV =  - ( it->weight * (2.0 * tmpSigVal - 1.0) * it->y);                                             
                            break;
                        }
                        vsArray[j][it->idx] = vsArray[j][it->idx] - gammat * stepV;
                    }
                    normalizeLength( vsArray[j] );
                    sigmoidOffSets[j] = tmpSigmoidOffSet;
                    sigmoidSlopes[j] = tmpSigmoidSlopes;
                                        
                }       
                // decrease gammat
                gammat = gammat / gammaDivider;
            }                                               
        } else if (_gMethod == OPT_BGD )
        {                       
            AlphaReal gammaDivider = 1.0;                   
                        
            for (int gradi=0; gradi<_maxIter; ++gradi)
            {
                if ((gradi>0)&&((gradi%_gammdivperiod)==0)) gammaDivider += 1.0;
                for (int j = 0; j < numColumns; ++j)
                {                                                                               
                    AlphaReal slopeImporvemement = 0.0;
                    AlphaReal offsetImporvemement = 0.0;
                    vector<AlphaReal> vImprovement(numClasses);
                    fill(vImprovement.begin(),vImprovement.end(),0.0);
                                        
                                        
                    switch (_tFunction) {
                    case TF_EXPLOSS:                                                        
                        for (int i = 0; i < numExamples; ++i )                                  
                        {
                            vector<Label> labels = _pTrainingData->getLabels(i);
                            FeatureReal val = _pTrainingData->getValue(i,j);
                                                                
                                                                
                            // update the parameters                                                
                            FeatureReal tmpSigVal = sigmoid(val,sigmoidSlopes[j],sigmoidOffSets[j]);
                                                                
                            AlphaReal deltaQ = 0.0;
                            for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                            {
                                deltaQ += exp( -vsArray[j][it->idx] * it->y * ( 2*tmpSigVal-1 ) )
                                    * 2.0 * it->weight*vsArray[j][it->idx]*it->y*tmpSigVal*(1.0-tmpSigVal);
                            }
                                                                
                            offsetImporvemement -= deltaQ;
                            slopeImporvemement -= (deltaQ*val);
                                                                
                                                                
                            for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                            {
                                // + gradient since it a maximization task
                                vImprovement[it->idx] -= exp( -vsArray[j][it->idx] * it->y * ( 2*tmpSigVal-1 ) ) 
                                    * ( it->weight * (2.0 * tmpSigVal - 1.0) * it->y);
                            }
                                                                
                        }                                                               
                        break;
                    case TF_EDGE:
                        for (int i = 0; i < numExamples; ++i )                                  
                        {
                            vector<Label> labels = _pTrainingData->getLabels(i);
                            FeatureReal val = _pTrainingData->getValue(i,j);
                                                                
                                                                
                            // update the parameters                                                
                            FeatureReal tmpSigVal = sigmoid(val,sigmoidSlopes[j],sigmoidOffSets[j]);
                                                                
                            AlphaReal deltaQ = 0.0;
                            for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                            {
                                deltaQ += 2.0 * it->weight*vsArray[j][it->idx]*it->y*tmpSigVal*(1.0-tmpSigVal);
                            }
                                                                
                            offsetImporvemement -= deltaQ;
                            slopeImporvemement -= (deltaQ*val);
                                                                
                                                                
                            for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                            {
                                // + gradient since it a maximization task
                                vImprovement[it->idx] -= ( it->weight * (2.0 * tmpSigVal - 1.0) * it->y);
                            }
                                                                
                        }                                                               
                        break;
                    }
                                                                                                                        
                    //update the current parameter vector
                    for( int iimp=0; iimp < vImprovement.size(); ++iimp )
                    {
                        // + gradient since it a maximization task                                              
                        vsArray[j][iimp] -= gammat * vImprovement[iimp];
                        //vsArray[j][iimp] += 100.0 * gammat * vImprovement[iimp];
                    }
                    normalizeLength( vsArray[j] );
                    sigmoidOffSets[j] -= gammat * offsetImporvemement;
                    //sigmoidOffSets[j] += 100.0 * gammat * offsetImporvemement;
                    sigmoidSlopes[j] -= gammat * slopeImporvemement;
                    //sigmoidSlopes[j] += 100.0 * gammat * slopeImporvemement;
                                        
                    // decrease gammat
                    gammat = gammat / gammaDivider;
                                        
                } // j 
            }
        } else {
            cout << "Unknown optimization method!" << endl;
            exit(-1);
        }
                
        int bestColumn = -1;
                
        // find the best feature
        for (int j = 0; j < numColumns; ++j)
        {
            _selectedColumn = j;
            _sigmoidSlope = sigmoidSlopes[j];
            _sigmoidOffset = sigmoidOffSets[j];
            _v = vsArray[j];
                        
            for(int k=0; k<numClasses; ++k ) _v[k]= _v[k] < 0 ? -1.0 : 1.0;
                        
            AlphaReal tmpEdge = this->getEdge();                    
                        
            if ((tmpEdge>0.0) && (tmpEdge>bestEdge))
            {
                bestEdge = tmpEdge;                             
                bestv = _v;
                bestColumn = j;
            }
        }
                
        _selectedColumn = bestColumn;
                
        if (_verbose>3) cout << "Selected column: " << _selectedColumn << endl;
                
        if ( _selectedColumn != -1 )
        {
            stringstream parameterString;
            parameterString << _sigmoidSlope << "_" << _sigmoidOffset;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + _id + parameterString.str();                      
        } else {
            return numeric_limits<AlphaReal>::signaling_NaN();
        }                                               
                
        _sigmoidSlope = sigmoidSlopes[_selectedColumn];
        _sigmoidOffset = sigmoidOffSets[_selectedColumn];
        _v = bestv;
        //normalizeLength( _v );
                
        if (_verbose>3)
        {
            cout << "Sigmoid slope:\t" << _sigmoidSlope << endl;
            cout << "Sigmoid offset:\t" << _sigmoidOffset << endl;                  
        }
                
                
        //calculate alpha
        this->_alpha = 0.0;
        AlphaReal eps_min = 0.0, eps_pls = 0.0;
                
        //_pTrainingData->clearIndexSet();
        for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {
            vector< Label> l = _pTrainingData->getLabels( i );
            for( vector< Label >::iterator it = l.begin(); it != l.end(); it++ ) {
                AlphaReal result  = this->classify( _pTrainingData, i, it->idx );
                result *= (it->y * it->weight);
                if ( result < 0 ) eps_min -= result;
                if ( result > 0 ) eps_pls += result;
            }
                        
        }
                
        this->_alpha = getAlpha( eps_min, eps_pls );
                
        // calculate the energy (sum of the energy of the leaves
        //AlphaReal energy = this->getEnergy( eps_min, eps_pls );
                
        return bestEdge;                                                        
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SigmoidSingleStumpLearner::run( int colIdx )
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
                
                
        const pair<vpIterator,vpIterator> dataBeginEnd = 
            static_cast<SortedData*>(_pTrainingData)->getFilteredBeginEnd( colIdx );
                
                
        const vpIterator dataBegin = dataBeginEnd.first;
        const vpIterator dataEnd = dataBeginEnd.second;
                
        bestEnergy = getEnergy(mu, tmpAlpha, tmpV);
                
        _alpha = tmpAlpha;
        _v = tmpV;
        _selectedColumn = colIdx;
                
        if ( _selectedColumn != -1 )
        {
            stringstream parameterString;
            parameterString << _sigmoidSlope << "_" << _sigmoidOffset;
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + _id + parameterString.str();                      
        } else {
            bestEnergy = numeric_limits<float>::signaling_NaN();
        }
                
        return bestEnergy;
                
    }
        
    // ------------------------------------------------------------------------------
    AlphaReal SigmoidSingleStumpLearner::run( vector<int>& colIndexes )
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
                
        AlphaReal bestEnergy = numeric_limits<AlphaReal>::signaling_NaN();
                
        return bestEnergy;
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SigmoidSingleStumpLearner::phi(FeatureReal val ) const
    {
                
        if (val > _sigmoidOffset)
            return -1;
        else
            return +1;
                
//              return static_cast<AlphaReal>(-2.0 / ( 1.0 + exp((double)(-(_sigmoidSlope*val + _sigmoidOffset) ) ) ) - 1.0);
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SigmoidSingleStumpLearner::phi(InputData* pData,int pointIdx) const
    {
        return phi(pData->getValue(pointIdx,_selectedColumn),0);
    }
        
    // -----------------------------------------------------------------------
        
    void SigmoidSingleStumpLearner::save(ofstream& outputStream, int numTabs)
    {
        // Calling the super-class method
        FeaturewiseLearner::save(outputStream, numTabs);
                
        // save selectedCoulumn
        outputStream << Serialization::standardTag("sigSlope", _sigmoidSlope, numTabs) << endl;
        outputStream << Serialization::standardTag("sigOffset", _sigmoidOffset, numTabs) << endl;               
                
    }
        
    // -----------------------------------------------------------------------
        
    void SigmoidSingleStumpLearner::load(nor_utils::StreamTokenizer& st)
    {
        // Calling the super-class method
        FeaturewiseLearner::load(st);
                
        _sigmoidSlope = UnSerialization::seekAndParseEnclosedValue<FeatureReal>(st, "sigSlope");
        _sigmoidOffset = UnSerialization::seekAndParseEnclosedValue<FeatureReal>(st, "sigOffset");
                
        stringstream parameterString;
        parameterString << _sigmoidSlope << "_" << _sigmoidOffset;
        _id = _id + parameterString.str();
    }
        
    // -----------------------------------------------------------------------
        
    void SigmoidSingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        FeaturewiseLearner::subCopyState(pBaseLearner);
                
        SigmoidSingleStumpLearner* pSigmoidSingleStumpLearner =
            dynamic_cast<SigmoidSingleStumpLearner*>(pBaseLearner);
                
        pSigmoidSingleStumpLearner->_sigmoidSlope = _sigmoidSlope;
        pSigmoidSingleStumpLearner->_sigmoidOffset = _sigmoidOffset;            
    }
        
    // -----------------------------------------------------------------------
    void SigmoidSingleStumpLearner::normalizeLength( vector<AlphaReal>& vec )
    {
        AlphaReal sum = 0.0;
        for(vector<AlphaReal>::iterator it = vec.begin(); it != vec.end(); ++it )
        {
            sum += ((*it)*(*it));                   
        }
        if (nor_utils::is_zero(sum)) return;
                
        sum = sqrt((double)sum);                
                
        for(int i=0; i<vec.size(); ++i)
        {
            vec[i]/=sum;                    
        }               
    }
    // -----------------------------------------------------------------------      
        
    AlphaReal SigmoidSingleStumpLearner::update( int trainingInstanceIdx )
    {               
        const int numColumns = _pTrainingData->getNumAttributes();                                                              
                        
        _age++;
        // compute nu
        AlphaReal nu = 1.0 / (double) (_age * _lambda); // regularized          
                
        AlphaReal edgeForCurrentInstance = 0.0;
                                                
        AlphaReal stepOffSet, stepSlope, stepV;
        vector<Label> labels = _pTrainingData->getLabels(trainingInstanceIdx);                          

        // calculate the edges before update the parameters
        for (int j = 0; j < numColumns; ++j)
        {
            FeatureReal val = _pTrainingData->getValue(trainingInstanceIdx, j);
            FeatureReal sig = sigmoid(val,_sigmoidSlopes[j],_sigmoidOffSets[j]);
                        
                        
            for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
            {
                AlphaReal deltaEdge = it->weight * it->y * sig * _vsArray[j][it->idx];
                _edges[j] += deltaEdge;
                _sumEdges[j] += (deltaEdge>0)?deltaEdge:-deltaEdge;
            }
        }

        for (int j = 0; j < numColumns; ++j)
        {                       
            FeatureReal val = _pTrainingData->getValue(trainingInstanceIdx, j);                                     
            FeatureReal tmpSigVal = sigmoid(val,_sigmoidSlopes[j],_sigmoidOffSets[j]);                      
                        
            AlphaReal sig = sigmoid(val,_sigmoidSlopes[j],_sigmoidOffSets[j]);
            AlphaReal scaledSigmoid = 2*sig-1;
            AlphaReal partialSigmoid = sig * (1.0 - sig);
                        
            AlphaReal deltaQ = 0.0;
                        
            switch (_tFunction) {
            case TF_EXPLOSS:
                for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                {
                    AlphaReal expLoss = exp(-1.0 * _vsArray[j][it->idx] * scaledSigmoid * it->y);
                    deltaQ += expLoss * 2.0 * it->weight*_vsArray[j][it->idx]*it->y*partialSigmoid;
                                                
                    stepV = expLoss * it->weight * scaledSigmoid * it->y;
                    _vsArray[j][it->idx] = _vsArray[j][it->idx] - _gammat * stepV;                                          
                    //_vsArray[j][it->idx] = (1.0 - 1.0 / _age) * _vsArray[j][it->idx] + nu * stepV; 
                }
                                                                                
                stepOffSet = -deltaQ;
                stepSlope  = -deltaQ * val;                                                     
                break;
            case TF_EDGE:
                for( vector< Label >::iterator it = labels.begin(); it != labels.end(); it++ )
                {
                    deltaQ += 2.0 * it->weight*_vsArray[j][it->idx]*it->y*partialSigmoid;
                                                
                    stepV =  - ( it->weight * scaledSigmoid * it->y);       
                    _vsArray[j][it->idx] = _vsArray[j][it->idx] + _gammat * stepV;
                }
                                        
                // because edge should be maximized
                stepOffSet = deltaQ;
                stepSlope  = deltaQ * val;
                break;
            default:
                break;
            }
                                                                                                
            normalizeLength( _vsArray[j] );
            // gradient step
            FeatureReal tmpSigmoidOffSet = _sigmoidOffSets[j] - static_cast<FeatureReal>(_gammat * stepOffSet);                                                                             
            FeatureReal tmpSigmoidSlopes = _sigmoidSlopes[j] - static_cast<FeatureReal>(_gammat * stepSlope);                                                                                                                       
//                      FeatureReal tmpSigmoidOffSet = (1.0 - 1.0 / _age) * _sigmoidOffSets[j] + static_cast<FeatureReal>(nu * stepOffSet);                                                                             
//                      FeatureReal tmpSigmoidSlopes = (1.0 - 1.0 / _age) * _sigmoidSlopes[j] + static_cast<FeatureReal>(nu * stepSlope);                                                                                                                       

                        
            _sigmoidOffSets[j] = tmpSigmoidOffSet;
            _sigmoidSlopes[j] = tmpSigmoidSlopes;
                                        
        }                                       
                
        // decrease gammat
        if ((_age>0)&&((_age%_gammdivperiod)==0)) _gammaDivider += 1.0;
        _gammat = _gammat / _gammaDivider;              
                
        return edgeForCurrentInstance;                                                  
    }
    // -----------------------------------------------------------------------      
    void SigmoidSingleStumpLearner::initLearning()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();                                              
                                
        _v.resize(numClasses);
        _gammat = _initialGammat;
                
        _sigmoidSlopes.resize( numColumns );
        _sigmoidOffSets.resize( numColumns );           
        _edges.resize( numColumns );
        _sumEdges.resize( numColumns );
                
        fill( _sigmoidSlopes.begin(), _sigmoidSlopes.end(), 0.1 );
        fill( _sigmoidOffSets.begin(), _sigmoidOffSets.end(), 0.0 );
        fill( _edges.begin(), _edges.end(), 0.0 );
        fill( _sumEdges.begin(), _sumEdges.end(), 0.0 );
                
        _vsArray.resize(numColumns);            
        for(int i=0; i<numColumns; ++i ) 
        {
            _vsArray[i].resize(numClasses);
            for (int k=0; k<numClasses; ++k )
            {
                _vsArray[i][k] = (rand() % 2) * 2 - 1;
            }
            normalizeLength( _vsArray[i] );
            //fill(vsArray[i].begin(),vsArray[i].end(),0.0);
        }
                                
        _gammaDivider = 1.0;    
        _age = 0;
    }

    // -----------------------------------------------------------------------      
    AlphaReal SigmoidSingleStumpLearner::finishLearning()
    {
        if (_verbose>3)
        {
                        
            const int numColumns = _pTrainingData->getNumAttributes();                                              
            for (int j = 0; j < numColumns; ++j)
            {
                _selectedColumn = j;
                _sigmoidSlope = _sigmoidSlopes[j];
                _sigmoidOffset = _sigmoidOffSets[j];
                _v = _vsArray[j];
                                
                //for(int k=0; k<numClasses; ++k ) _v[k]= _v[k] < 0 ? -1.0 : 1.0;
                                
                if (_verbose>4)
                {
                    AlphaReal tmpEdge = this->getEdge(true);                        
                    cout << "--------> " << j << " " << tmpEdge << " " << _edges[j] / _sumEdges[j] << endl;
                }
            }                       
        }               
                
        // the best column has to be chosen
        AlphaReal bestEdge = -numeric_limits<FeatureReal>::max();
        for( int i = 0; i < _edges.size(); ++i )
        {
            AlphaReal currEdges = _edges[i] / _sumEdges[i];
            if (bestEdge<currEdges)
            {
                bestEdge = currEdges;
                _selectedColumn = i;
            }
        }
                
        _sigmoidSlope = _sigmoidSlopes[_selectedColumn];
        _sigmoidOffset = _sigmoidOffSets[_selectedColumn];
        _v = _vsArray[_selectedColumn];
                
        for( int i = 0; i < _v.size(); ++i )
            _v[i] = (_v[i]<0)?-1:1;
                
        // release the memory used during the training
        _sigmoidSlopes.resize(0);
        _sigmoidOffSets.resize(0);
        _vsArray.resize(0);
        _edges.resize(0);
        _sumEdges.resize(0);
                
                
        if (_verbose>3)
        {
            cout << "Selected column: " << _selectedColumn << endl;
            cout << "Best edge :\t" << bestEdge << endl;
            cout << "Sigmoid slope:\t" << _sigmoidSlope << endl;
            cout << "Sigmoid offset:\t" << _sigmoidOffset << endl;                  
        }
                
                
        return bestEdge;
    }

} // end of namespace MultiBoost
