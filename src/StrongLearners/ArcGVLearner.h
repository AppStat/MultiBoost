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
 * \file ArcGVLearner.h The meta-learner arc-gv.
 * See 
 */
#pragma warning( disable : 4786 )

#ifndef __ARCGV_LEARNER_H
#define __ARCGV_LEARNER_H

#include "StrongLearners/AdaBoostMHLearner.h"
#include "Utils/Args.h"
#include "Defaults.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
    /**
     * The arc-gv learner. This class performs the meta-learning
     * by calling the weak learners and updating the weights. The
     * main difference between AdaBoost.MH and arc-gv is that the 
     * weight of weak classifier is regularized with the minimal margin.
     * See Breiman, L. (1998). Arcing classifiers. The Annals of Statistics, 26, 801–-849.
     * \date 04/04/2012
     */
    class ArcGVLearner : public AdaBoostMHLearner
    {
    public:
        
        /**
         * The constructor. It initializes the variables and sets them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \date 13/11/2005
         */
    ArcGVLearner() : AdaBoostMHLearner(), _alphaSum(0.0), _minMarginThreshold( 0.0 ) {}
        
        /**
         * Start the learning process.
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \see OutputInfo
         * \date 10/11/2005
         */
        virtual void run(const nor_utils::Args& args);                          
    protected:
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
                
                
    private:
        /**
         * Fake assignment operator to avoid warning.
         * \date 04/04/2012
         */
        AdaBoostMHLearner& operator=( const AdaBoostMHLearner& ) {return *this;}
        
        AlphaReal _alphaSum;
        AlphaReal _minMarginThreshold;
    };
    
} // end of namespace ArcGV

#endif // __ARCGV_LEARNER_H
