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
 * \file ConstantLearner.h A single threshold decision stump learner. 
 */

#ifndef __CONSTANT_LEARNER_H
#define __CONSTANT_LEARNER_H

#include "WeakLearners/AbstainableLearner.h"
#include "WeakLearners/ScalarLearner.h"
#include "WeakLearners/StochasticLearner.h"

#include "Utils/Args.h"
#include "IO/InputData.h"

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
    class ConstantLearner : public virtual AbstainableLearner, public virtual ScalarLearner, public virtual StochasticLearner
    {
    public:
                                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~ConstantLearner() {}
                
        /**
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 14/11/2005
         */
        virtual BaseLearner* subCreate() { return new ConstantLearner(); }

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
        virtual void declareArguments(nor_utils::Args& args) 
        {
            AbstainableLearner::declareArguments( args );
        }
                
                
        /**
         * Set the arguments of the algorithm using the standard interface
         * of the arguments. Call this to set the arguments asked by the user.
         * \param args The arguments defined by the user in the command line.
         * \date 24/04/2007
         */
        virtual void initLearningOptions(const nor_utils::Args& args) 
        {
            AbstainableLearner::initLearningOptions( args );
        }
                
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
         * Run the learner to build the classifier on the given data.
         * \param pData The pointer to the data.
         * \see BaseLearner::run
         * \date 11/11/2005
         */
        virtual AlphaReal run();
                
        // TODO: comment
        virtual AlphaReal run( int colNum );
                
        /**
         * Returns a vector of float holding any data that the specific weak learner can generate
         * using the given input dataset. Right now just a single case is contemplated, therefore
         * "reason" is not used. The returned vector of data correspond to:
         * \f[
         * \left\{ v^{(t)}_1, v^{(t)}_2, \cdots, v^{(t)}_K, \phi^{(t)}({\bf x}_1), 
         * \phi^{(t)}({\bf x}_2), \cdots, \phi^{(t)}({\bf x}_n) \right\}
         * \f]
         * where \f$v\f$ is the alignment vector, \f$\phi\f$ is the discriminative function and
         * \f$t\f$ is the current iteration (to which this instantiation of weak learner belongs to).
         * \param data The vector with the returned data.
         * \param pData The pointer to the data points (the examples) previously loaded.
         * \remark This particular method has been created for analysis purposes (research). If you
         * need to get similar analytical information from this weak learner, add it to the function
         * and uncomment the parameter "reason".
         * \see BaseLearner::getStateData
         * \see Classifier::saveSingelStumpFeatureData
         * \date 10/2/2006
         */
        virtual void getStateData( vector<FeatureReal>& data, const string& /*reason = ""*/, InputData* pData = 0 );
                
                
        /**
         * Initializes the parameters for online learning. This comes from the StochasticLearner InterFace.              
         */
        virtual void initLearning();
                
        /**
         * Release memory used during training.
         * \date 03/11/2011
         */                             
        virtual AlphaReal finishLearning();
                
        /**
         * It updates the parameter of the constant learner in online learning.
         * \return edge for the given istance
         * \date 03/11/2011
         */
        virtual AlphaReal update( int idx );
                
                
                
    protected:
                
        /**
         * A discriminative function. 
         * \remarks Positive or negative do NOT refer to positive or negative classification.
         * This function is equivalent to the phi function in my thesis.
         * \param val The value to discriminate
         * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
         * \return +1 always 
         * \date 19/09/2006
         * \see classify
         */
        virtual AlphaReal phi(InputData* pData, int idx, int classIdx) const { return 1; }
                
        virtual AlphaReal cut(  InputData* pData, int idx ) const { return 1; }                         
    };
        
    //////////////////////////////////////////////////////////////////////////
        
} // end of namespace MultiBoost

#endif // __CONSTANT_LEARNER_H
