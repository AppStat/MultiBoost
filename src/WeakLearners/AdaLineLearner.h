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
 * \file AdaLineLearner.h The implementation of adaline learner. The weak learner has the form of \bv \bw^T \bx
 */

#ifndef __ADALINE_LEARNER_H
#define __ADALINE_LEARNER_H

#include "NeuronLearner.h"
#include "StochasticLearner.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    /**
     */
    class AdaLineLearner : public virtual NeuronLearner, public virtual StochasticLearner
    {                               
    public:
                
        /**
         * The constructor.
         * \date 29/04/2010
         */             
    AdaLineLearner() : NeuronLearner(), StochasticLearner() {}
                
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */             
        virtual ~AdaLineLearner() {}
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
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 21/05/2007
         */
        virtual BaseLearner* subCreate() { return new AdaLineLearner(); }
                
        /**
         * Run the learner to build the classifier on the given data.
         * \param pData The pointer to the data.
         * \see BaseLearner::run
         * \date 21/05/2007
         */
        virtual AlphaReal run();
                
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
         * It updates the parameter of weak learner.
         * \return edge for the given istance
         * \date 03/11/2011
         */
        virtual AlphaReal update( int idx );
                
    protected:              
        /**
         * A discriminative function. 
         * \param pData The data
         * \param idx The index of the data point
         * \param classIdx The class index
         * \return The output of the discriminant function 
         * \date 10/11/2010
         */
        virtual AlphaReal phi(InputData* pData, int idx, int classIdx) const
        {
            AlphaReal retval = 0.0;
            // inner product
            for(int i=0; i<_featuresWeight.size(); ++i )
            {
                retval += _featuresWeight[i] * pData->getValue(idx, i);
            }                       
                        
            return (retval<0.0)? 1.0 :-1.0; // signum
        }
                
        // variables for learning                                               
        vector<AlphaReal>                       _edges;
        vector<AlphaReal>                       _sumEdges;
                
                
    };
} // end of namespace MultiBoost

#endif // __ADALINE_LEARNER_H

