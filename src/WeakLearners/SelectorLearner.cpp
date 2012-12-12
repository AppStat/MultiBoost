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


#include "SelectorLearner.h"
#include <limits>

#include "IO/Serialization.h"

namespace MultiBoost {
        
    //REGISTER_LEARNER_NAME(SingleStump, SelectorLearner)

        
    // ------------------------------------------------------------------------------
        
    AlphaReal SelectorLearner::run()
    {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();
        const int numExamples = _pTrainingData->getNumExamples();
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( 1.0 / (AlphaReal)_pTrainingData->getNumExamples() * 0.01 );
                
        vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
                
        AlphaReal tmpAlpha, tmpEnergy;
        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
                
        int numOfDimensions = _maxNumOfDimensions;
        for (int j = 0; j < numColumns; ++j)
        {
            // Tricky way to select numOfDimensions columns randomly out of numColumns
            int rest = numColumns - j;
            float r = rand()/static_cast<float>(RAND_MAX);
                        
            if ( static_cast<float>(numOfDimensions) / rest > r ) 
            {
                --numOfDimensions;
                /*
                  if (_verbose > 2)
                  cout << "    --> trying attribute = "
                  <<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
                  << endl << flush;
                */
                const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();
                                
                // Create and initialize the numIdxs x numClasses gamma matrix
                vector<vector<AlphaReal> > tmpGammasPls(numIdxs);
                vector<vector<AlphaReal> > tmpGammasMin(numIdxs);
                for (int io = 0; io < numIdxs; ++io) {
                    vector<AlphaReal> tmpGammaPls(numClasses);
                    vector<AlphaReal> tmpGammaMin(numClasses);
                    fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
                    fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
                    tmpGammasPls[io] = tmpGammaPls;
                    tmpGammasMin[io] = tmpGammaMin;
                }
                                
                // Compute the elements of the gamma plus and minus matrices
                AlphaReal entry;
                for (int i = 0; i < numExamples; ++i) {
                    const vector<Label>& labels = _pTrainingData->getLabels(i);
                    int io = static_cast<int>(_pTrainingData->getValue(i,j));           
                    for (int l = 0; l < numClasses; ++l) {
                        entry = labels[l].weight * labels[l].y;
                        if (entry > 0)
                            tmpGammasPls[io][l] += entry;
                        else if (entry < 0)
                            tmpGammasMin[io][l] += -entry;
                    }
                }
                                
                // Initialize the u vector to random +-1
                vector<sRates> uMu(numIdxs); // The idx-wise rates
                vector<AlphaReal> tmpU(numIdxs);// The idx-wise votes/abstentions
                vector<AlphaReal> previousTmpU(numIdxs);// The idx-wise votes/abstentions
                                
                for (int io = 0; io < numIdxs; ++io) {
                    uMu[io].classIdx = io;  
                }
                                
                for (int io = 0; io < numIdxs; ++io) {                                  
                    // initializing u as it has only one positive element
                    fill( tmpU.begin(), tmpU.end(), -1 );
                    tmpU[io] = +1;
                                        
                    vector<sRates> vMu(numClasses); // The label-wise rates
                    for (int l = 0; l < numClasses; ++l)
                        vMu[l].classIdx = l;
                    vector<AlphaReal> tmpV(numClasses); // The label-wise votes/abstentions
                                        
                    AlphaReal tmpVal;
                    tmpAlpha = 0.0;
                                        
                    //filling out tmpV and vMu
                    for (int l = 0; l < numClasses; ++l) {
                        vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
                        for (int io = 0; io < numIdxs; ++io) {
                            if (tmpU[io] > 0) {
                                vMu[l].rPls += tmpGammasPls[io][l];
                                vMu[l].rMin += tmpGammasMin[io][l];
                            }
                            else if (tmpU[io] < 0) {
                                vMu[l].rPls += tmpGammasMin[io][l];
                                vMu[l].rMin += tmpGammasPls[io][l];
                            }
                        }
                        if (vMu[l].rPls >= vMu[l].rMin) {
                            tmpV[l] = +1;
                        }
                        else {
                            tmpV[l] = -1;
                            tmpVal = vMu[l].rPls;
                            vMu[l].rPls = vMu[l].rMin;
                            vMu[l].rMin = tmpVal;
                        }
                    }
                                        
                    tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);
                                        
                    if ( tmpEnergy < bestEnergy && tmpAlpha > 0 ) {
                        _alpha = tmpAlpha;
                        _v = tmpV;
                        //_u = tmpU;
                        _positiveIdxOfArrayU = io;
                        _selectedColumn = j;
                        bestEnergy = tmpEnergy;
                    }
                }
            }
        }
                
                
        if (_selectedColumn>-1)
        {
            _id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
            return bestEnergy;
        } else {
            return bestEnergy = numeric_limits<float>::signaling_NaN();
        }                               
    }
        
    // ------------------------------------------------------------------------------
    // TODO: implementing this function
    AlphaReal SelectorLearner::run( int colIdx )
    {
        cerr << "The featurewise run of SelectorLearner isn't ready!!!" << endl;
        exit(-1);
        return 0.0;
                
    }
        
        
    // ------------------------------------------------------------------------------
        
    AlphaReal SelectorLearner::phi(FeatureReal val) const
    {
        if (_positiveIdxOfArrayU == static_cast<int>(val) ) {
            return 1.0;
        } else {
            return -1.0;
        }
        //return _u[static_cast<int>(val)];
    }
        
    // -----------------------------------------------------------------------
        
    void SelectorLearner::save(ofstream& outputStream, int numTabs)
    {
        // Calling the super-class method
        FeaturewiseLearner::save(outputStream, numTabs);
                
        // sparse save the _u vector
        outputStream << Serialization::standardTag("positiveIdxOfU", _positiveIdxOfArrayU, numTabs) << endl;
    }
        
    // -----------------------------------------------------------------------
        
    void SelectorLearner::load(nor_utils::StreamTokenizer& st)
    {
        // Calling the super-class method
        FeaturewiseLearner::load(st);
                
        // load sparse phiArray data
        _positiveIdxOfArrayU = UnSerialization::seekAndParseEnclosedValue<int>(st, "positiveIdxOfU");
    }
        
    // -----------------------------------------------------------------------
        
    void SelectorLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        FeaturewiseLearner::subCopyState(pBaseLearner);
                
        SelectorLearner* pSelectorLearner =
            dynamic_cast<SelectorLearner*>(pBaseLearner);
                
        pSelectorLearner->_positiveIdxOfArrayU = _positiveIdxOfArrayU;
    }
        
} // end of namespace MultiBoost
