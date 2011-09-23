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



#include "NeuronLearner.h"
#include "IO/Serialization.h"

namespace MultiBoost {
        
    // ------------------------------------------------------------------------------
        
    void NeuronLearner::declareArguments(nor_utils::Args& args)
    {
        AbstainableLearner::declareArguments(args);
    }
        
    // ------------------------------------------------------------------------------
        
    void NeuronLearner::initLearningOptions(const nor_utils::Args& args)
    {
        AbstainableLearner::initLearningOptions(args);
    }
        
    // -----------------------------------------------------------------------
        
    void NeuronLearner::save(ofstream& outputStream, int numTabs)
    {
        // Calling the super-class method
        AbstainableLearner::save(outputStream, numTabs);
                
        // save feature weights         
        stringstream featureWeightsSS("");              
        vector<FeatureReal>::iterator it = _featuresWeight.begin();
        featureWeightsSS << *it;
        ++it;
        for (;it != _featuresWeight.end(); ++it )
        {
            featureWeightsSS << " " << *it;
        }
                
        outputStream << Serialization::standardTag("fweights", featureWeightsSS.str(), numTabs) << endl;
    }
        
    // -----------------------------------------------------------------------
        
    void NeuronLearner::load(nor_utils::StreamTokenizer& st)
    {
        // Calling the super-class method
        AbstainableLearner::load(st);
                
        _featuresWeight.resize(0);
                
        // load feature weights
        locale tmpLocale  = locale(locale(), new nor_utils::white_spaces(", "));
        string featureWeightsString = UnSerialization::seekAndParseEnclosedValue<string>(st, "fweights");
                
        stringstream featureWeightsSS(featureWeightsString);
        featureWeightsSS.imbue(tmpLocale);
                
        // store the weights
        while (!featureWeightsSS.eof())
        {
            FeatureReal tmpWeight;
            featureWeightsSS >> tmpWeight;
                        
            _featuresWeight.push_back( tmpWeight );
        }
                
                
    }
        
    // -----------------------------------------------------------------------      
    void NeuronLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        AbstainableLearner::subCopyState(pBaseLearner);
                
        NeuronLearner* pNeuronLearner =
            dynamic_cast<NeuronLearner*>(pBaseLearner);
                
        pNeuronLearner->_featuresWeight = _featuresWeight;
    }
        
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
