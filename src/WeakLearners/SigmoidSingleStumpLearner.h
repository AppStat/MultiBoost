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
 * \file SingleStumpLearner.h A single threshold decision stump learner. 
 */

#ifndef __SIGMOID_SINGLE_STUMP_LEARNER_H
#define __SIGMOID_SINGLE_STUMP_LEARNER_H

#include "WeakLearners/StochasticLearner.h"
#include "WeakLearners/ScalarLearner.h"
#include "WeakLearners/FeaturewiseLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"
#include "IO/SortedData.h"

#include <vector>
#include <fstream>
#include <cassert>
#include <math.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    /**
     * A \b single threshold decision stump learner. 
     * There is ONE and ONE ONLY threshold here.
     */
    class SigmoidSingleStumpLearner : public virtual FeaturewiseLearner, public virtual ScalarLearner, public virtual StochasticLearner
    {
    public:
                
    SigmoidSingleStumpLearner() : _sigmoidSlope(numeric_limits<FeatureReal>::signaling_NaN()),
            _sigmoidOffset(numeric_limits<FeatureReal>::signaling_NaN())
            {}

        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~SigmoidSingleStumpLearner() {}
                
        /**
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 14/11/2005
         */
        virtual BaseLearner* subCreate() { return new SigmoidSingleStumpLearner(); }
                
                
                
        /**
         * It updates the parameter of weak learner.
         * \return edge for the given istance
         * \date 03/11/2011
         */
        virtual AlphaReal update( int idx );
                                
        /**
         * Allocate memory for training.
         * \date 03/11/2011
         */             
        virtual void initLearning();
                
        /**
         * Release memory usied during training.
         * \date 03/11/2011
         */                             
        virtual AlphaReal finishLearning();                             
                
        /**
         * Creates an InputData object that it is good for the
         * weak learner. Overridden to return SortedData.
         * \see InputData
         * \see BaseLearner::createInputData()
         * \see SortedData
         * \warning The object \b must be destroyed by the caller.
         * \date 21/11/2005
         */
        virtual InputData* createInputData() { return new InputData(); }
                
        /**
         * Declare weak-learner-specific arguments.
         * These arguments will be added to the list of arguments under 
         * the group specific of the weak learner. It is called
         * automatically in main, when the list of arguments is built up.
         * Use this method to declare the arguments that belongs to
         * the weak learner only.
         * 
         * This class declares the argument "rtrainingsize".
         * \param args The Args class reference which can be used to declare
         * additional arguments.
         * \date 19/07/2006
         */
        virtual void declareArguments(nor_utils::Args& args);
                
        /**
         * Set the arguments of the algorithm using the standard interface
         * of the arguments. Call this to set the arguments asked by the user.
         * \param args The arguments defined by the user in the command line.
         * \date 24/04/2007
         */
        virtual void initLearningOptions(const nor_utils::Args& args);
                
        /**
         * Run the learner to build the classifier on the given data.
         * \param pData The pointer to the data.
         * \see BaseLearner::run
         * \date 11/11/2005
         */
        virtual AlphaReal run();
                
        virtual AlphaReal run( int colIdx );
                
        virtual AlphaReal run( vector<int>& colIndexes );
                
        /**
         * Save the current object information needed for classification,
         * that is the single threshold.
         * \param outputStream The stream where the data will be saved
         * \param numTabs The number of tabs before the tag. Useful for indentation
         * \remark To fully save the object it is \b very \b important to call
         * also the super-class method.
         * \see StumpLearner::save()
         * \date 13/11/2005
         */
        virtual void save(ofstream& outputStream, int numTabs = 0);
                
        /**
         * Load the xml file that contains the serialized information
         * needed for the classification and that belongs to this class.
         * \param st The stream tokenizer that returns tags and values as tokens
         * \see save()
         * \date 13/11/2005
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
                                
        /**
         * The same discriminative function as below, but called with a data point. 
         * Called only from HierarchicalStumpLearner::phi
         * \param pData The input data.
         * \param pointIdx The index of the data point.
         * \return +1 if \a pData[pointIdx][_selectedColumn] is on one side of the 
         * border for \a classIdx and -1 otherwise.
         * \date 17/02/2006
         */
        virtual AlphaReal phi(InputData* pData, int pointIdx) const;
                
    protected:
                
        inline FeatureReal sigmoid( FeatureReal val, FeatureReal sigSlope = 1.0, FeatureReal sigOffset = 0.0 )
        {
            FeatureReal retVal = 1.0 / ( 1.0 + exp((double)(-(sigSlope*val + sigOffset) ) ) );
            return retVal;
        }
                
        void normalizeLength( vector<AlphaReal>& vec );
                
        /**
         * A discriminative function. 
         * \remarks Positive or negative do NOT refer to positive or negative classification.
         * This function is equivalent to the phi function in my thesis.
         * \param val The value to discriminate
         * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
         * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
         * \date 11/11/2005
         * \see classify
         */
        virtual AlphaReal phi(FeatureReal val ) const;
                
        /**
         * A discriminative function. 
         * \remarks Positive or negative do NOT refer to positive or negative classification.
         * This function is equivalent to the phi function in my thesis.
         * \param val The value to discriminate
         * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
         * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
         * \date 11/11/2005
         * \see classify
         */             
        virtual AlphaReal phi(FeatureReal val, int classIdx) const { return phi(val); }
                
        /**
         * A simple cut on a feature. This simple function is needed for TreeLearner.
         * \param pData The input data.
         * \param idx The index of the data point.
         * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
         * \date 11/11/2005
         */             
        virtual AlphaReal cut( InputData* pData, int idx ) const
        {
            return phi( pData->getValue( idx, _selectedColumn) );
        }                               
                
        FeatureReal _sigmoidSlope; //!< the slope parameter of sigmoid
        FeatureReal _sigmoidOffset; //!< the offset parameter of sigmoid                
                
        // data variables for training
        vector<FeatureReal>                     _sigmoidSlopes;
        vector<FeatureReal>                     _sigmoidOffSets;
                
        vector<vector<AlphaReal> >      _vsArray;
        vector<AlphaReal>                       _edges;
        vector<AlphaReal>                       _sumEdges;
    };
        
    //////////////////////////////////////////////////////////////////////////
        
} // end of namespace MultiBoost

#endif // __SINGLE_STUMP_LEARNER_H
