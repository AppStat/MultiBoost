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
 * \file FeaturewiseLearner.h It represents all weak learners that search all or
 * or a subset of features. and implement phi(x,l) as phi(x[selectedFeature],l)
 */

#ifndef __FEATUREWISE_LEARNER_H
#define __FEATUREWISE_LEARNER_H

#include "WeakLearners/AbstainableLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"

#include <vector>
#include <fstream>
#include <limits>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    /**
     * A generic featurewise learner. It represents all weak learners that search all or
     * or a subset of features, and implement phi(x,l) as phi(x[selectedFeature],l)
     *
     * \date 19/07/2006
     */
    class FeaturewiseLearner : public virtual AbstainableLearner
    {
    public:
                
        /**
         * The constructor. It initializes the selected column to -1.
         * \date 19/07/2006
         */
    FeaturewiseLearner() : _selectedColumn(-1), _maxNumOfDimensions( numeric_limits<int>::max() ) {}
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~FeaturewiseLearner() {}
                
                
        /**
         * Run the learner to build the classifier using only one single feature from the given data.
         * \param pData The pointer to the data.
         * \warning This function \b must update _alpha too! You can use the
         * helper functions (the getAlpha with parameters) to update it.
         * \params colIdx The index of the feature can be only used.
         * \return The energy of the weak classifier (that we want to minimize)
         * \see getAlpha(float)
         * \see getAlpha(float, float)
         * \see getAlpha(float, float, float, float)
         */             

        virtual AlphaReal run( int colIdx ) = 0;
                
        /**
         * Run the learner to build the classifier using only a subset of features from the given data.
         * \param pData The pointer to the data.
         * \warning This function \b must update _alpha too! You can use the
         * helper functions (the getAlpha with parameters) to update it.
         * \params colIndices The indices of the features can be only used.
         * \return The energy of the weak classifier (that we want to minimize)
         * \see getAlpha(float)
         * \see getAlpha(float, float)
         * \see getAlpha(float, float, float, float)
         */             
        // TODO: /*# should be implemented #*/
        virtual AlphaReal run( vector<int>& colIndices );
                
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
         * \date 19/07/2006
         */
        virtual void declareArguments(nor_utils::Args& args);
                
        /**
         * Set the arguments of the algorithm using the standard interface
         * of the arguments. Call this to set the arguments asked by the user.
         * \param args The arguments defined by the user in the command line.
         * \date 14/11/2005
         * \remark These options are used for training only!
         */
        virtual void initLearningOptions(const nor_utils::Args& args);
                
        /**
         * Save the current object information needed for classification,
         * that is: \a _selectedColumn, the column of the 
         * data with that yielded the lowest error
         * \param outputStream The stream where the data will be saved
         * \param numTabs The number of tabs before the tag. Useful for indentation
         * \remark To fully save the object it is \b very \b important to call
         * also the super-class method.
         * \see AbstainableLearner::save()
         * \date 13/11/2005
         */
        virtual void save(ofstream& outputStream, int numTabs = 0);
                
        /**
         * Load the xml file that contains the serialized information
         * needed for the classification and that belongs to this class
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
         * A discriminative function. 
         * \param pData The data
         * \param idx The index of the data point
         * \param classIdx The class index
         * \return The output of the discriminant function 
         * \date 19/07/2006
         */
        virtual AlphaReal phi(InputData* pData, int idx, int classIdx) const;
                
        /**
         * A scalar discriminative function. 
         * \param val The value to discriminate, it is interpreted in the _selectedColumn'th feature
         * \param classIdx The class index
         * \return 
         * \date 11/11/2005
         */
        virtual AlphaReal phi(FeatureReal val, int classIdx) const = 0;
                
        int    _selectedColumn; //!< The column of the training data with the lowest error.
        int    _maxNumOfDimensions; //!< limit on the number of searched dimensions in run()
    };
        
    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------
        
} // end of namespace MultiBoost

#endif // __FEATUREWISE_LEARNER_H
