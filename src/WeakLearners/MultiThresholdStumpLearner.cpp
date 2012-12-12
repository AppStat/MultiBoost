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

#include "MultiThresholdStumpLearner.h"

#include "IO/Serialization.h"
#include "IO/SortedData.h"

#include "Algorithms/StumpAlgorithm.h"

#include <limits> // for numeric_limits<>
namespace MultiBoost {



    // ------------------------------------------------------------------------------

    AlphaReal MultiThresholdStumpLearner::run() {
        const int numClasses = _pTrainingData->getNumClasses();
        const int numColumns = _pTrainingData->getNumAttributes();

        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal(1.0 / (AlphaReal) _pTrainingData->getNumExamples() * 0.01);

        vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.

        vector<AlphaReal> tmpV(numClasses); // The class-wise votes/abstentions
        vector<FeatureReal> tmpThresholds(numClasses);
        AlphaReal tmpAlpha;

        AlphaReal bestEnergy = numeric_limits<AlphaReal>::max();
        AlphaReal tmpEnergy;

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
                    static_cast<SortedData*> (_pTrainingData)->getFilteredBeginEnd(
                        j);

                const vpIterator dataBegin = dataBeginEnd.first;
                const vpIterator dataEnd = dataBeginEnd.second;

                sAlgo.findMultiThresholdsWithInit(dataBegin, dataEnd,
                                                  _pTrainingData, tmpThresholds, &mu, &tmpV);

                tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
                if (tmpEnergy < bestEnergy && tmpAlpha > 0) {
                    // Store it in the current algorithm
                    // note: I don't really like having so many temp variables
                    // but the alternative would be a structure, which would need
                    // to be inheritable to make things more consistent. But this would
                    // make it less flexible. Therefore, I am still undecided. This
                    // might change!

                    _alpha = tmpAlpha;
                    _v = tmpV;
                    _selectedColumn = j;
                    _thresholds = tmpThresholds;

                    bestEnergy = tmpEnergy;
                }
            }
        }

        return bestEnergy;

    }

// ------------------------------------------------------------------------------

    AlphaReal MultiThresholdStumpLearner::phi(FeatureReal val, int classIdx) const {
        if (val > _thresholds[classIdx])
            return +1;
        else
            return -1;
    }

// -----------------------------------------------------------------------

    void MultiThresholdStumpLearner::save(ofstream& outputStream, int numTabs) {
        // Calling the super-class method
        FeaturewiseLearner::save(outputStream, numTabs);

        // save all the thresholds
        outputStream << Serialization::vectorTag("thArray", _thresholds,
                                                 _pTrainingData->getClassMap(), "class", static_cast<FeatureReal>(0.0), numTabs)
                     << endl;
    }

// -----------------------------------------------------------------------

    void MultiThresholdStumpLearner::load(nor_utils::StreamTokenizer& st) {
        // Calling the super-class method
        FeaturewiseLearner::load(st);

        // load vArray data
        UnSerialization::seekAndParseVectorTag(st, "thArray",
                                               _pTrainingData->getClassMap(), "class", _thresholds);
    }

// -----------------------------------------------------------------------

    void MultiThresholdStumpLearner::subCopyState(BaseLearner *pBaseLearner) {
        FeaturewiseLearner::subCopyState(pBaseLearner);

        MultiThresholdStumpLearner* pMultiThresholdStumpLearner =
            dynamic_cast<MultiThresholdStumpLearner*> (pBaseLearner);

        pMultiThresholdStumpLearner->_thresholds = _thresholds;
    }

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
