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
 * \file MultiStumpLearner.h A multi threshold decision stump learner.
 */

#ifndef __MULTI_STUMP_LEARNER_H
#define __MULTI_STUMP_LEARNER_H

#include "WeakLearners/FeaturewiseLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"
#include "IO/SortedData.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    /**
     * A \b multi threshold decision stump learner.
     * There is a threshold for every class.
     */
    class MultiStumpLearner: public AbstainableLearner {
    public:
                
        /**
         * The constructor. It initializes the array of selected columns to be empty.
         * \date 15/07/2010
         */
                
    MultiStumpLearner() :
        AbstainableLearner(), _selectedColumnArray(0) {
        }
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of
         * the object.
         */
        virtual ~MultiStumpLearner() {
        }
                
        /**
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 14/11/2005
         */
        virtual BaseLearner* subCreate() {
            return new MultiStumpLearner();
        }
                
        /**
         * Creates an InputData object that it is good for the
         * weak learner. Overridden to return SortedData.
         * \see InputData
         * \see BaseLearner::createInputData()
         * \see SortedData
         * \warning The object \b must be destroyed by the caller.
         * \date 21/11/2005
         */
        virtual InputData* createInputData() {
            return new SortedData();
        }
                
        /**
         * Run the learner to build the classifier on the given data.
         * \param pData The pointer to the data
         * \see BaseLearner::run
         * \date 11/11/2005
         */
        virtual AlphaReal run();
                
        /**
         * Save the current object information needed for classification,
         * that is the threshold list.
         * \param outputStream The stream where the data will be saved
         * \param numTabs The number of tabs before the tag. Useful for indentation
         * \remark To fully save the object it is \b very \b important to call
         * also the super-class method.
         * \see StumpLearner::save()
         * \date 13/11/2005
         */
                
        /**
         * Declare weak-learner-specific arguments.
         * These arguments will be added to the list of arguments under 
         * the group specific of the weak learner. It is called
         * automatically in main, when the list of arguments is built up.
         * Use this method to declare the arguments that belongs to
         * the weak learner only.
         * 
         * This class declares the argument "rsample" only.
         * \param args The Args class reference which can be used to declare
         * additional arguments.
         * \date 03/11/2010
         */
        virtual void declareArguments(nor_utils::Args& args);
                
        /**
         * Set the arguments of the algorithm using the standard interface
         * of the arguments. Call this to set the arguments asked by the user.
         * \param args The arguments defined by the user in the command line.
         * \date 03/11/20010
         * \remark These options are used for training only!
         */
        virtual void initLearningOptions(const nor_utils::Args& args);
                
                
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
                
    protected:
        /**
         * A discriminative function. This function has to be overloaded here,
         * because the MultiStumpLearner carries out the classification based
         * on different feature for each class.
         * \param pData The data
         * \param idx The index of the data point
         * \param classIdx The class index
         * \return The output of the discriminant function
         * \date 19/07/2006
         */
        virtual AlphaReal phi(InputData* pData, int idx, int classIdx) const;
                
        /**
         * A discriminative function.
         * \remarks Positive or negative do NOT refer to positive or negative classification.
         * This function is equivalent to the phi function in my thesis.
         * \param val The value to discriminate
         * \param classIdx The index of the class
         * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
         * \date 11/11/2005
         * \see classify
         */
        virtual AlphaReal phi(FeatureReal val, int classIdx) const;
                
        vector<int> _selectedColumnArray; // The selected columns having the smallest error for each class separately
        vector<FeatureReal> _thresholds; //!< The thresholds (one for each class) of the decision stump.
        int    _maxNumOfDimensions; //!< limit on the number of searched dimensions in run()
    };
        
    //////////////////////////////////////////////////////////////////////////
        
} // end of namespace MultiBoost

#endif // __MULTI_STUMP_LEARNER_H
