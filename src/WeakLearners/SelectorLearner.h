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


/**
 * \file SelectorLearner.h A single value indicator learner. 
 */

#ifndef __SELECTOR_LEARNER_H
#define __SELECTOR_LEARNER_H

#include "FeaturewiseLearner.h"
#include "ScalarLearner.h"
#include "Utils/Args.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    /**
     * A \b single threshold decision stump learner. 
     * There is ONE and ONE ONLY threshold here.
     */
    class SelectorLearner : public FeaturewiseLearner, public ScalarLearner
    {
    public:
                
    SelectorLearner() : FeaturewiseLearner(), ScalarLearner(), _positiveIdxOfArrayU(-1) { }
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~SelectorLearner() {}
                
        /**
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 21/05/2007
         */
        virtual BaseLearner* subCreate() { return new SelectorLearner(); }
                
        /**
         * Run the learner to build the classifier on the given data.
         * \see BaseLearner::run
         * \date 21/05/2007
         */
        virtual AlphaReal run();

        /**
         * Run the learner to build the classifier using only one single feature from the given data.            
         * \see FeaturewiseLearner::run
         * \date 21/05/2009
         */             
        virtual AlphaReal run( int colIdx );
                
        /**
         * Save the current object information needed for classification,
         * that is the _u vector.
         * \param outputStream The stream where the data will be saved
         * \param numTabs The number of tabs before the tag. Useful for indentation
         * \remark To fully save the object it is \b very \b important to call
         * also the super-class method.
         * \date 21/05/2007
         */
        virtual void save(ofstream& outputStream, int numTabs = 0);
                
        /**
         * Load the xml file that contains the serialized information
         * needed for the classification and that belongs to this class.
         * \param st The stream tokenizer that returns tags and values as tokens
         * \see save()
         * \date 21/05/2007
         */
        virtual void load(nor_utils::StreamTokenizer& st);
                
        /**
         * Copy all the info we need in classify().
         * pBaseLearner was created by subCreate so it has the correct (sub) type.
         * Usually one must copy the same fields that are loaded and saved. Don't 
         * forget to call the parent's subCopyState().
         * \param pBaseLearner The sub type pointer into which we copy.
         * \see save
         * \see load
         * \see classify
         * \see ProductLearner::run()
         * \date 25/05/2007
         */
        virtual void subCopyState(BaseLearner *pBaseLearner);
                
    protected:
        /**
         * A discriminative function. 
         * \remarks Positive or negative do NOT refer to positive or negative classification.
         * \param val The value to discriminate
         * \return \phi[(int)val]
         * \date 11/11/2005
         * \see classify
         */
        virtual AlphaReal phi(FeatureReal val ) const;
                
        /**
         * A scalar discriminative function. 
         * \param val The value to discriminate, it is interpreted in the _selectedColumn'th feature
         * \param classIdx The class index
         * \return 
         * \date 11/11/2005
         */
        virtual AlphaReal phi(FeatureReal val, int classIdx) const { return phi(val); }
                
        /**
         * A simple cut on a feature. This simple function is needed for TreeLearner.
         * \param pData The input data.
         * \param idx The index of the data point.
         * \return +1 if the correspinding value of pData[idx][_selectedColumn] in array _u
         * is positive and -1 otherwise.
         * \date 21/07/2010
         */
        virtual AlphaReal cut( InputData* pData, int idx ) const
        {
            return phi( pData->getValue( idx, _selectedColumn) );
        }
                                
        int                       _positiveIdxOfArrayU;         
    };
        
    //////////////////////////////////////////////////////////////////////////
        
} // end of namespace MultiBoost

#endif // __ENUM_LEARNER_H
