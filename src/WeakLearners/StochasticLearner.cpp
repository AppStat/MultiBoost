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


#include "WeakLearners/StochasticLearner.h"

namespace MultiBoost {
        
    // ------------------------------------------------------------------------------
        
    void StochasticLearner::declareArguments(nor_utils::Args& args)
    {
        BaseLearner::declareArguments(args);
        args.declareArgument("graditer",
                             "Declares the number of randomly drawn training size for SGD"
                             "whereas it declares the number of iteration for the Batch Gradiend Descend"                                                    
                             " size <num> of training set. "
                             "Example: --graditer 50 -> Uses only 50 randomly chosen training instance",
                             1, "<num>");
                
        args.declareArgument("gradmethod",
                             "Declares the gradient method: "
                             " (sgd) Stochastic Gradient Descent, (bgd) Batch Gradient Descent"
                             "Example: --gradmethod sgd -> Uses stochastic gradient method",
                             1, "<method>");
                
        args.declareArgument("tfunc",
                             "Target function: "
                             "exploss: Exponential Loss, edge: max. edge"
                             "Example: --tfunc exploss -> Uses exponantial loss for minimizing",
                             1, "<function>");
                
        args.declareArgument("initgamma",
                             "The initial learning rate in gradient descent"
                             "Default values is 10.0",
                             1, "<gamma>");
                
        args.declareArgument("gammdivperiod",
                             "The periodicity of decreasing the learning rate \\gamma"
                             "Default values is 1",
                             1, "<period>");
                
                                
    }
        
    // ------------------------------------------------------------------------------
        
    void StochasticLearner::initLearningOptions(const nor_utils::Args& args)
    {
        BaseLearner::initLearningOptions(args);
                
        if (args.hasArgument("initgamma"))
            args.getValue("initgamma", 0, _initialGammat);                  
                
        if (args.hasArgument("gammdivperiod"))
            args.getValue("gammdivperiod", 0, _gammdivperiod);              
                
                
        if (args.hasArgument("graditer"))
            args.getValue("graditer", 0, _maxIter);                 
                
        if (args.hasArgument("gradmethod"))
        {
            string gradMethod;
            args.getValue("gradmethod", 0, gradMethod);             
                        
            if ( gradMethod.compare( "sgd" ) == 0 )
                _gMethod = OPT_SGD;
            else if ( gradMethod.compare( "bgd" ) == 0 )
                _gMethod = OPT_BGD;
            else {
                cerr << "SigmoidSingleStumpLearner::Unknown update gradient method" << endl;
                exit( -1 );
            }                                       
        }               
                
        if (args.hasArgument("tfunc"))
        {
            string targetFunction;
            args.getValue("tfunc", 0, targetFunction);
                        
            if ( targetFunction.compare( "exploss" ) == 0 )
                _tFunction = TF_EXPLOSS;
            else if ( targetFunction.compare( "edge" ) == 0 )
                _tFunction = TF_EDGE;
            else {
                cerr << "SigmoidSingleStumpLearner::Unknown target function" << endl;
                exit( -1 );                             
            }                                       
                        
        }
                
    }
        
        
    // -----------------------------------------------------------------------
        
    void StochasticLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        BaseLearner::subCopyState(pBaseLearner);
                                
        StochasticLearner* pStochasticLearner =
            dynamic_cast<StochasticLearner*>(pBaseLearner);
                
                
        pStochasticLearner->_gMethod = _gMethod;
        pStochasticLearner->_maxIter = _maxIter;
        pStochasticLearner->_tFunction = _tFunction;
                
                
        pStochasticLearner->_gammat           = _gammat;                
        pStochasticLearner->_gammaDivider = _gammaDivider;
        pStochasticLearner->_age              = _age;
        pStochasticLearner->_initialGammat= _initialGammat;
        pStochasticLearner->_nu                   = _nu;
        pStochasticLearner->_lambda               = _lambda;            
        pStochasticLearner->_gammdivperiod= _gammdivperiod;             
                
    }
                
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
