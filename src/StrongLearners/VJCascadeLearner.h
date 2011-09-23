
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
 * \file VJCascadeLearner.h The Viola-Jones cascade learner, for further details see:
 \verbatim
 @Article{ViJo01,
 author =       {Viola, P. and Jones, M.},
 title =        {Robust Real-time Face Detection},
 journal =      {International Journal of Computer Vision},
 year =         {2004},
 volume =       {57},
 pages =        {137--154}
 } \endverbatim
 *
 */

#ifndef __VJ_CASCADE_LEARNER_H
#define __VJ_CASCADE_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    struct CascadeOutputInformation {
        bool    active;
        int             forecast;
        int             classifiedInStage;
        int             numberOfUsedClassifier;
        double  score;
    };
        
    /**
     * The AdaBoost learner. This class performs the meta-learning
     * by calling the weak learners and updating the weights.
     * \date 12/11/2005
     */
    class VJCascadeLearner : public GenericStrongLearner
    {
    public:
                
        /**
         * The constructor. It initializes the variables and sets them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \date 13/11/2005
         */
    VJCascadeLearner()
        : _numIterations(0), _maxTime(-1), _verbose(1), _smallVal(1E-10), _stageStartNumber(2),
            _resumeShypFileName(""), _outputInfoFile(""), _withConstantLearner(false),
            _maxAcceptableFalsePositiveRate(0.6), _minAcceptableDetectionRate(0.99),
            _positiveLabelName(""), _positiveLabelIndex(0), _thresholds(0), _outputPosteriorsFileName(""),
            _testFileName(""), _validFileName("") {}

        /**
         * Declare argumets specific to VJCascade algorithm.
         * \param args The Args class reference which can be used to declare
         * additional arguments.
         * \date 02/08/2011
         */
        static void declareBaseArguments(nor_utils::Args& args) {
            args.setGroup("Viola-Jones Cascade Algorithm Options");
            args.declareArgument("firstStage", "[optional] The number of weak classifier in the first stage", 1, "<val>" );
            args.declareArgument("positivelabel", "The name of positive label", 1, "<labelname>" );
            args.declareArgument("minacctpr", "The minimum acceptabel detection rate/TPR, see. VJ paper Table 2. par. d (default 0.99)", 1, "<val>" );
            args.declareArgument("maxaccfpr", "The maximum acceptabel FPR, see. VJ paper Table 2. par. f (default 0.6)", 1, "<val>" );
            args.declareArgument("stagewiseposteriors", "[optional] Output the stagewiseposteriors", 1, "<fname>" );
        }
                
        /**
         * Start the learning process.
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \see OutputInfo
         * \date 10/11/2005
         */
        virtual void run(const nor_utils::Args& args);
                
        /**
         * Performs the classification using the AdaBoostMHClassifier.
         * \param args The arguments provided by the command line with all
         * the options for classification.
         */
        virtual void classify(const nor_utils::Args& args);
                
        /**
         * Print to stdout (or to file) a confusion matrix.
         * \param args The arguments provided by the command line.
         * \date 20/3/2006
         */
        virtual void doConfusionMatrix(const nor_utils::Args& args);
                
        /**
         * Output the outcome of the strong learner for each class.
         * Strictly speaking these are (currently) not posteriors,
         * as the sum of these values is not one.
         * \param args The arguments provided by the command line.
         */
        virtual void doPosteriors(const nor_utils::Args& args);
                                                
        /**
         * In every stage ut updates the posteriors of the training and validation data.
         */
        static void calculatePosteriors( InputData* pData, vector<BaseLearner*>& weakhyps, vector<AlphaReal>& posteriors, int positiveLabelIndex );
                
    protected:
        /**
         * The same weight update like in AdaBoost.MH, but note that the weights of the instances
         * are set to the proportional weighting scheme at the begining of each stage.
         * \see AdaBoostMHLearner::updateWeights()
         * \param pTrainingData The pointer to the training data.
         * \param pWeakHypothesis The current weak hypothesis.
         * \return The value of the edge. It will be used to see if the algorithm can continue
         * with learning.
         * \date 13/07/2011
         */
        virtual AlphaReal updateWeights(InputData* pData, BaseLearner* pWeakHypothesis);
                
                
        /**
         * Resets he weights of the instances.
         * At the begining of each stage, the weights of training instances
         * has to be set according to the proportional weighting.
         * \param pTrainingData The pointer to the training data.
         * \date 13/07/2011
         */             
        virtual void resetWeights(InputData* pData);
                
        /**
         * Updates the posteriors after each stage.
         * \params pData The pointer to the data.
         * \params weakhyps The array of the weak learners found in the current stage.
         * \params posteriors The array of posterios.
         * \params positiveLabelIndex 
         * \date 13/07/2011
         */
        static void updatePosteriors( InputData* pData, BaseLearner* weakhyps, vector<AlphaReal>& posteriors, int positiveLabelIndex );
                
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param args The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
                
                
        /**
         * Resume the weak learner list.
         * \return The current iteration number. 0 if not -resume option has been called
         * \param pTrainingData The pointer to the training data, needed for classMap, enumMaps.
         * \date 21/12/2005
         * \see resumeProcess
         * \remark resumeProcess must be called too!
         */
        int resumeWeakLearners(InputData* pTrainingData);
                
                
        /**
         * Auxiliary function for checking whether the sum of the weight is equal to zero.
         * \params pData The pointer to the data.
         * \date 13/07/2011              
         */
        void checkWeights( InputData* pData );
                
        /**
         * Calculates the true positive rate and false negative rate for a given dataset with a given threshold.
         * \params pData The pointer to the data. The instances above the threshold will be classified as positive.
         * \params TPR The true positive rate.
         * \params FPR The false positive rate.
         * \params threshold The threshold (by default it is equal to zero).
         * \date 13/07/2011
         */
        virtual void getTPRandFPR( InputData* pData, vector<AlphaReal>& posteriors, AlphaReal& TPR, AlphaReal& FPR, const FeatureReal threshold = 0.0 );
                
        /**
         * Calculates the threshold for a given TPR rate.
         * \params pData The pointer to the data. The instances above the threshold will be classified as positive.
         * \params TPR The true positive rate.
         * \params FPR The false positive rate.
         * \params threshold The threshold (by default it is equal to zero).
         * \date 13/07/2011
         */
        virtual FeatureReal getThresholdBasedOnTPR( InputData* pData, vector<AlphaReal>& posteriors, const AlphaReal expectedTPR, AlphaReal& TPR, AlphaReal& FPR );
                
                
        virtual void forecastOverAllCascade( InputData* pData, vector<AlphaReal>& posteriors, vector<CascadeOutputInformation>& cascadeData, const FeatureReal threshold );
                
                
        // for output
        virtual void outputHeader();
        virtual void outputCascadeResult( InputData* pData, vector<CascadeOutputInformation>& cascadeData );            
                
        vector<vector<BaseLearner*> >  _foundHypotheses; //!< The list of the hypotheses found.
        vector<FeatureReal>  _thresholds; //!< The list of the hypotheses found.
                
        string  _baseLearnerName; //!< The name of the basic learner used by AdaBoost. 
        string  _shypFileName; //!< File name of the strong hypothesis.
        bool       _isShypCompressed; 
                
        string  _trainFileName;
        string  _validFileName;
        string  _testFileName;
                
        ofstream _output;
        ofstream _outputPosteriors;
                
        string  _positiveLabelName; //!< the name of positive class e.g. face class in a face detection task
        int             _positiveLabelIndex; //!< the index of the positive class
                
        int     _numIterations;
        int     _maxTime; //!< Time limit for the whole processing. Default: no time limit (-1).                
        int             _stageStartNumber;
        /**
         * Verbose level.
         * There are three levels of verbosity:
         * - 0 = no messages
         * - 1 = basic messages
         * - 2 = show all messages
         */
        int     _verbose;
        const AlphaReal _smallVal; //!< A small value, to solve numeric issues
                
        /**
         * If resume is set, this will hold the strong hypothesis file to load in order to 
         * continue with the training process.
         */
        string  _resumeShypFileName;
        string  _outputInfoFile; //!< The filename of the step-by-step information file that will be updated
        string  _outputPosteriorsFileName;
                
        bool    _withConstantLearner;
                
        double _maxAcceptableFalsePositiveRate; //! this variable is denoted by f in the Viola-Jones paper
        double _minAcceptableDetectionRate; //! this variable is denoted by d in the Viola-Jones paper
                
        vector<vector<AlphaReal> > _validTable;
        vector<vector<AlphaReal> > _testTable;          
        ////////////////////////////////////////////////////////////////
    private:
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        VJCascadeLearner& operator=( const VJCascadeLearner& ) {return *this;}
                
        /**
         * A temporary variable for h(x)*y. Helps saving time during re-weighting.
         */                             
        vector< vector<AlphaReal> > _hy;
    };
        
} // end of namespace MultiBoost

#endif // __ADABOOST_MH_LEARNER_H

