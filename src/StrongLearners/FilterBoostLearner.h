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


/**
 * \file FilterBoostLearner.h The meta-learner FilterBoost.
 */

#ifndef __FILTERBOOST_LEARNER_H
#define __FILTERBOOST_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "StrongLearners/AdaBoostMHLearner.h"
#include "Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

    /**
     * The FilterBoost learner. This class performs the meta-learning
     * by calling the weak learners and updating the weights.
     \verbatim
     @InProceedings{BrSc07,
     author =       {Bradley, J.K.  and Schapire, R.E.},
     title  =       {{FilterBoost}: Regression and Classification on Large
     Datasets},
     booktitle =    {Advances in Neural Information Processing Systems},
     volume =       {20},
     year =         {2008},
     publisher =    {The MIT Press}
     } \endverbatim
     *       
     * \date 04/07/2011
     */
    class FilterBoostLearner : public AdaBoostMHLearner
    {
    public:

        /**
         * The constructor. It initializes the variables and sets them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args. The constant learner is switched on by default.
         * \date 13/11/2005
         */
    FilterBoostLearner() : AdaBoostMHLearner(), _Cn(300), _onlineWeakLearning(false), _sumAlpha(0.0) {}

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

                
        /**
         * Resume the training using the features in _resumeShypFileName if the
         * option -resume has been specified. The margins should be updated as well.
         * \date 21/12/2005
         */
        void resumeProcess(Serialization& ss, InputData* pTrainingData, InputData* pTestData, 
                           OutputInfo* pOutInfo);
                
                
    private:
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        FilterBoostLearner& operator=( const FilterBoostLearner& ) {return *this;}

        /**
         * For margins
         */
        vector< vector<AlphaReal> > _margins;
                
        /**
         * The filter function. It draws random instances from the training data WITHOUT replacement.
         */
        void filter( InputData* pData, int size, bool rejection = true );
                                
        void setWeightToMargins( InputData* pData );
        void updateMargins( InputData* pData, BaseLearner* pWeakHypothesis );
        /**
         * The size of subset will be used for training the base learner
         */
        int _Cn;
                
        /**
         * The weak learner will be trained in an online fashion. In this case, GradientLearner has to be used.
         */
        bool _onlineWeakLearning;
        AlphaReal _sumAlpha;
        // temporary function
        void saveMargins();             
    };

} // end of namespace MultiBoost

#endif // __ADABOOST_MH_LEARNER_H

