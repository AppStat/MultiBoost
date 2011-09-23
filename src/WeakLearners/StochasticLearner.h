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


#ifndef __STOCHASTICLEARNER_H_
#define __STOCHASTICLEARNER_H_

#include "BaseLearner.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
    enum GradientMethods {
        OPT_SGD, // stochastic descend
        OPT_BGD  // batch gradient descend
    };
        
    enum TargetFunctions {
        TF_EDGE,
        TF_EXPLOSS
    };
        
        
    /**
     * \file StochasticLearner.h It represents all weak learners that can be trained in an online fashion.
     * \date 03/11/2011
     */
        
    class StochasticLearner : public virtual BaseLearner
    {
    public:         
    StochasticLearner() : BaseLearner(),
            _gMethod( OPT_BGD ),
            _tFunction( TF_EXPLOSS ),
            _maxIter(20),
            _gammaDivider(1.0),
            _initialGammat(10.0),
            _lambda(0.001),
            _gammdivperiod(1)
                
            {}
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of
         * the object.
         */
        virtual ~StochasticLearner() {}

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
         * Allocate memory for training.
         * \date 03/11/2011
         */             
        virtual void initLearning() = 0;

        /**
         * Release memory usied during training.
         * \date 03/11/2011
         */                             
        virtual AlphaReal finishLearning() = 0;
                
        /**
         * It updates the parameter of weak learner.
         * \return edge for the given istance
         * \date 03/11/2011
         */
        virtual AlphaReal update( int idx ) = 0;
    protected:
        GradientMethods            _gMethod;
        int                        _maxIter;
        TargetFunctions            _tFunction;
                
                
        AlphaReal                                       _gammat;                
        AlphaReal                                       _gammaDivider;
        int                                                     _age;
        AlphaReal                                       _initialGammat;
        AlphaReal                                       _nu;
        AlphaReal                                       _lambda;                
        int                                                     _gammdivperiod;         
    };
        
}
#endif /* SCALARLEARNER_H_ */
