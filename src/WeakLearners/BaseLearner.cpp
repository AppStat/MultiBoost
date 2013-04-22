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


#include <cmath> // for log and exp

#include "WeakLearners/BaseLearner.h"

#include "StrongLearners/AdaBoostMHLearner.h"
//#include "StrongLearners/BrownBoostLearner.h"
//#include "StrongLearners/LogitBoostLearner.h"
#include "StrongLearners/ArcGVLearner.h"
#include "StrongLearners/FilterBoostLearner.h"
#include "StrongLearners/VJCascadeLearner.h"
#include "StrongLearners/SoftCascadeLearner.h"

#include "Utils/Utils.h" // for is_zero

#include "IO/InputData.h"
#include "IO/Serialization.h" // for the helper function "standardTag"

namespace MultiBoost {

    // -----------------------------------------------------------------------------

    const     AlphaReal BaseLearner::_smallVal = 1e-3;
    int               BaseLearner::_verbose = 1;
    AlphaReal BaseLearner::_smoothingVal = BaseLearner::_smallVal;

    // -----------------------------------------------------------------------------

    void BaseLearner::declareArguments(nor_utils::Args& args)
    {
    }

    // -----------------------------------------------------------------------------

    void BaseLearner::declareBaseArguments(nor_utils::Args& args)
    {
        args.declareArgument("shypname", 
                             "The name of output strong hypothesis (default: "
                             + string(SHYP_NAME) + "." + string(SHYP_EXTENSION) + ").", 
                             1, "<filename>");

        args.declareArgument("shypcomp", 
                             "The shyp file will be compressed", 
                             1, "<flag 0-1>");

        args.setGroup("Basic Algorithm Options");
        args.declareArgument("resume", 
                             "Resumes a training process using the strong hypothesis file.", 
                             1, "<shypFile>");   
        args.declareArgument("edgeoffset", 
                             "Defines the value of the edge offset (theta) (default: no edge offset).", 
                             1, "<val>");        
    }

    // ------------------------------------------------------------------------------

    void BaseLearner::initLearningOptions(const nor_utils::Args& args)
    {
        if ( args.hasArgument("verbose") )
            args.getValue("verbose", 0, _verbose);

        // Set the value of theta
        if ( args.hasArgument("edgeoffset") )
            args.getValue("edgeoffset", 0, _theta);   
    }

    // -----------------------------------------------------------------------

    GenericStrongLearner* BaseLearner::createGenericStrongLearner( nor_utils::Args& args )
    {
        initLearningOptions(args);
        string sHypothesisName = "";
        if ( args.hasArgument("stronglearner") ) 
        {
            args.getValue("stronglearner", 0, sHypothesisName );
        } else
        {
            if ( _verbose > 0 ) 
            {
                cerr << "Warning: No strong learner is given. Set to default (AdaBoost)." << endl; 
            }
            sHypothesisName = "AdaBoostMH";
        }
        if ( _verbose > 0 ) 
        {
            cout << "The strong learner is " << sHypothesisName << endl; 
        }

        GenericStrongLearner* sHypothesis = NULL;

        if ( sHypothesisName.compare( "AdaBoostMH" ) == 0)
        {
            sHypothesis = new AdaBoostMHLearner();
        } else if ( sHypothesisName.compare( "FilterBoost" ) == 0 ) {
            sHypothesis = new FilterBoostLearner();
        } else if ( sHypothesisName.compare( "ArcGV" ) == 0 ) {
            sHypothesis = new ArcGVLearner();                       
        } else if ( sHypothesisName.compare( "VJcascade" ) == 0 ) {
            sHypothesis = new VJCascadeLearner();
        } else if ( sHypothesisName.compare( "SoftCascade") == 0) {
            sHypothesis = new SoftCascadeLearner();
        } else {
            cout << "Unknown strong learner!!!!" << endl;
            exit( -1 );
        }
        return sHypothesis;
    }

    // -----------------------------------------------------------------------

    InputData* BaseLearner::createInputData()
    {
        return new InputData();
    }

    // -----------------------------------------------------------------------

    AlphaReal BaseLearner::getAlpha(AlphaReal eps_min, AlphaReal eps_pls) const
    {
        return 0.5 * log( (eps_pls + _smoothingVal) / (eps_min + _smoothingVal) );
    }

    // -----------------------------------------------------------------------

    AlphaReal BaseLearner::getAlpha(AlphaReal eps_min, AlphaReal eps_pls, AlphaReal theta) const
    {
        // if theta == 0
        if ( nor_utils::is_zero(theta) )
            return getAlpha( eps_min, eps_pls );

        const AlphaReal eps_zero = 1 - eps_min - eps_pls;

        if (eps_min < _smallVal)
        {
            // if eps_min == 0
            return log( ( (1-theta)* eps_pls ) / (theta * eps_zero) );
        }
        else
        {
            // ln( -b + sqrt( b^2 + c) );
            const AlphaReal denom = (1+theta) * eps_min;
            const AlphaReal b = ((theta) * eps_zero) / (2*denom);
            const AlphaReal c = ((1-theta) * eps_pls) / denom;

            return log( -b + sqrt( b * b + c ) );
        }

    }

    // -----------------------------------------------------------------------

    AlphaReal BaseLearner::getEnergy(AlphaReal eps_min, AlphaReal eps_pls) const
    {
        return 2 * sqrt( eps_min * eps_pls ) + ( 1 - eps_min - eps_pls );
    }

    // -----------------------------------------------------------------------

    AlphaReal BaseLearner::getEnergy(AlphaReal eps_min, AlphaReal eps_pls, AlphaReal alpha, AlphaReal theta) const
    {
        // if theta == 0
        if ( nor_utils::is_zero(theta) )
            return getEnergy( eps_min, eps_pls );

        return exp(alpha * theta) * ( eps_min * exp(alpha) +  eps_pls * exp(-alpha)
                                      +  (1 - eps_min - eps_pls) );

    }

    // -----------------------------------------------------------------------

    void BaseLearner::save(ofstream& outputStream, int numTabs)
    {
        // save name
        outputStream << Serialization::standardTag("weakLearner", _name, numTabs) << endl;

        // save alpha
        outputStream << Serialization::standardTag("alpha", _alpha, numTabs) << endl;
    }

    // -----------------------------------------------------------------------

    void BaseLearner::load(nor_utils::StreamTokenizer& st)
    {
        // name should be loaded by caller so he knows what derived class to load
        // He will then call create() which copies the name

        // load alpha
        _alpha = UnSerialization::seekAndParseEnclosedValue<AlphaReal>(st, "alpha");
    }

    // -----------------------------------------------------------------------

    BaseLearner* BaseLearner::copyState()
    {
        BaseLearner *pBaseLearner = subCreate();
        subCopyState(pBaseLearner);
        return pBaseLearner;
    }

    // -----------------------------------------------------------------------

    void BaseLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        pBaseLearner->_theta = _theta;
        pBaseLearner->_alpha = _alpha;
        pBaseLearner->_name = _name;
        pBaseLearner->_id = _id;
        pBaseLearner->_pTrainingData = _pTrainingData;
    }

    // -----------------------------------------------------------------------

    AlphaReal BaseLearner::getEdge( bool isNormalized )
    {
        AlphaReal edge = 0.0;
        AlphaReal sumPos = 0.0;
        AlphaReal sumNeg = 0.0;

        for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {

            vector< Label > l = _pTrainingData->getLabels( i );
            //cout << d->getRawIndex( i ) << " " << endl;

            for( vector<Label>::iterator it = l.begin(); it !=  l.end(); it++ ) {
                AlphaReal cl = classify( _pTrainingData, i, it->idx );
                AlphaReal tmpVal = cl * it->weight * it->y;
                if ( tmpVal >= 0.0 ) sumPos += tmpVal;
                else sumNeg -= tmpVal;
            }
        }
        //cout << endl;
                
        edge = sumPos - sumNeg;
                
        if ( isNormalized )
        {
            //if ( _pTrainingData->isFiltered() ) {
            AlphaReal sumEdge = sumNeg + sumPos;
            if ( ! nor_utils::is_zero( sumEdge ) ) edge /= sumEdge; 
            //} 
        }
        return edge;
    }
    // -----------------------------------------------------------------------


} // end of namespace MultiBoost
