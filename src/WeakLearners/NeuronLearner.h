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
 * \file NeuronLearner.h It represents all weak learners can be written
 * in the form of \bv^{T}\varphi (\bw^\bx)
 */

#ifndef __NEURON_LEARNER_H
#define __NEURON_LEARNER_H

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
     * \date 10/11/2010
     */
    class NeuronLearner : public virtual AbstainableLearner
    {
    public:
                
        /**
         * The constructor. It initializes the selected column to -1.
         * \date 10/11/2010
         */
    NeuronLearner() : AbstainableLearner(), _featuresWeight(0) {}
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~NeuronLearner() {}
                
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
         * \date 10/11/2010
         */
        virtual void declareArguments(nor_utils::Args& args);
                
        /**
         * Set the arguments of the algorithm using the standard interface
         * of the arguments. Call this to set the arguments asked by the user.
         * \param args The arguments defined by the user in the command line.
         * \date 10/11/2010
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
         * \date 10/11/2010
         */
        virtual void save(ofstream& outputStream, int numTabs = 0);
                
        /**
         * Load the xml file that contains the serialized information
         * needed for the classification and that belongs to this class
         * \param st The stream tokenizer that returns tags and values as tokens
         * \see save()
         * \date 10/11/2010
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
         * \date 10/11/2010
         */
        virtual void subCopyState(BaseLearner *pBaseLearner);
                
    protected:
                
        vector<FeatureReal> _featuresWeight; //!< The weight of the features.
    };
        
    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------
        
} // end of namespace MultiBoost

#endif // __NEURON_LEARNER_H

