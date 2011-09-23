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
 * \file SoftCascade.h 
 */
#pragma warning( disable : 4786 )

#ifndef __SOFT_CASCADE_LEARNER_H
#define __SOFT_CASCADE_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
    
    class BaseLearner;
    class InputData;
    class Serialization;
    class OutputInfo;
    
    /**
     * The SoftCascade learner. From "Robust object detection via soft cascade"
     * This learner takes a strong hypothesis file (by default shyp.xml) as input
     * and outputs a embedded-cascade like classifier.
     * \link http://www.lubomir.org/academic/softcascade.pdf
     * \date 01/07/2011
     */
    class SoftCascadeLearner : public GenericStrongLearner
    {
    public:
        
        /**
         * The constructor.
         * \date 01/07/2011
         */
    SoftCascadeLearner()
        : _numIterations(0), _verbose(1), _smallVal(1E-10),
            _withConstantLearner(false), _sepWidth(12), _trainPosteriorsFileName(""), _testPosteriorsFileName(""), _fullRun(false), _inShypLimit(0), _outputInfoFile("") 
            , _bootstrapRate(0), _bootstrapFileName( "" ), _alphaExponentialParameter(0.0), _targetDetectionRate(0.95) {}
        
        
        /**
         * Declare argumets specific to SoftCascade algorithm.
         * \param args The Args class reference which can be used to declare
         * additional arguments.
         * \date 01/07/2011
         */
        static void declareBaseArguments(nor_utils::Args& args) {
            args.setGroup("SoftCascade Algorithm Options");
            args.declareArgument("positivelabel", "The name of positive label", 1, "<labelname>" );
            args.declareArgument("detectionrate", "The target detection rate (true positive rate)", 1, "<real>" );
            args.declareArgument("expalpha", "The parameter of the exponential distribution (speed/accuracy trade-off, alpha < 0 for speed and alpha > 0 for accuracy)", 1, "<alpha>" );
            args.declareArgument("calibrate", "The shyp file of the already trained classifier, if not given, AdaBoost.MH will be run before the SoftCascade to generate a shyp file.", 1, "<file>" );
            args.declareArgument("calibrate", "The shyp file of the already trained classifier, <number> limits the number of weak hypotheses read.", 2, "<file> <number>" );
            args.declareArgument("bootstrap", "[optional] bootstrap K% negatives at each iteration from the bootstrap dataset", 2, "<bootstrap file name> <K>" );
            args.declareArgument("trainposteriors", "[optional] The name of the train posteriors output file", 1, "<file>" );
            args.declareArgument("testposteriors", "[optional] The name of the test posteriors output file", 1, "<file>" );

        }
        
        /**
         * Start the learning process.
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \date 10/11/2005
         */
        virtual void run(const nor_utils::Args& args);
        
        /**
         * Performs the classification using the SoftCascadelassifier.
         * \param args The arguments provided by the command line with all
         * the options for classification.
         */
        virtual void classify(const nor_utils::Args& args);
        
        
        /**
         * Output the outcome of the strong learner for each class.
         * Strictly speaking these are (currently) not posteriors,
         * as the sum of these values is not one.
         * \param args The arguments provided by the command line.
         */
        virtual void doPosteriors(const nor_utils::Args& args);
        
        /**
         * Print to stdout (or to file) a confusion matrix.
         * \param args The arguments provided by the command line.
         * \date 01/07/2011
         */
        // TODO: to be implemented
        virtual void doConfusionMatrix(const nor_utils::Args& args) {  };                                
        
    protected:
        
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
         */
        void getArgs(const nor_utils::Args& args);
        
        /**
         * Wrapper function to avoid raw/indirect index problems
         * \date 01/07/2011
         */
        int getInstanceLabel(InputData* pData, int i, int positiveLabelIndex) const;
        
        /**
         * Initialize the per-stage rejection allowance of true positives
         * \param iNumberOfStages The target number of stages (weak classifiers) of the cascade
         * \param oVector The output vector of rejection allowance
         * \date 01/07/2011
         */
        void initializeRejectionDistributionVector(const int iNumberOfStages, vector<double>& oVector);
       
        /**
         * Compute the balanced edge of a set of weak learners.
         * \param iPosteriors The input posteriors vector of the set of weak learners 
         * \date 01/07/2011
         */
        AlphaReal computeSeparationSpan(InputData* pData, const vector<AlphaReal> & iPosteriors, int positiveLabelIndex);
        
        /**
         * Compute the posteriors (the score of the strong learner, sometimes called the predictor) from a
         * set of weak hypotheses
         * \date 01/07/2011
         */
        void computePosteriors(InputData* pData, vector<BaseLearner*> & weakHypotheses, vector<AlphaReal> & oPosteriors, int positiveLabelIndex);
        
        /**
         * Upate the posteriors after the addition of a weak hypothesis
         * \date 01/07/2011
         */
        void updatePosteriors( InputData* pData, BaseLearner* weakHypotheses, vector<AlphaReal>& oPosteriors, int positiveLabelIndex);
        
        /**
         * Find the rejection threshold that satisfes the rejection distribution vector while discarding the most possible negatives.
         * (see the paper)
         * \param faceRejectionFraction The amount of positives that we permit to discard.
         * \date 01/07/2011
         */
        AlphaReal findBestRejectionThreshold(InputData* pData, const vector<AlphaReal> & iPosteriors, const double & faceRejectionFraction, double & oMissesFraction);
        
        /**
         * Filter the examples that are below the rejection threshold stage-wise.
         * \param indices The set of example indices that are still used for training
         * \date 01/07/2011
         */
        int filterDataset(InputData* pData, const vector<AlphaReal> & posteriors, AlphaReal threshold, set<int> & indices);
        
        /**
         * Output the header to the output file 
         * \date 01/07/2011
         */
        void outputHeader();
        
        /**
         * Ouput the cascade performance (error etc.)
         * \param outScores A rather general vector that contains some performance information on the examples
         * \date 01/07/2011
         */
        void outputCascadePerf(InputData* pData, vector< vector< AlphaReal> > & outScores);
        
        /**
         * Add sampled negatives to the training set in order to create new false positive
         * (see the paper)
         * \param pData The training dataset
         * \param pBootData A training set containing only negatives from which false positives are sampled
         * \date 01/07/2011
         */
        void bootstrapTrainingSet(InputData * pData, InputData * pBootData, set<int> & indices);

        /**
         * Print output information if option --outputinfo is specified.
         * \date 04/07/2011
         */
        void printOutputInfo(OutputInfo* pOutInfo, int t, 
                             InputData* pTrainingData, InputData* pTestData, 
                             BaseLearner* pWeakHypothesis,
                             AlphaReal r);
        /**
         * Update the posteriors table inside OutputInfo.
         * This design was choosen because the cascade structure 
         * necessitate a particular posterior updating.
         * \date 04/07/2011
         */

//        void updateOutputInfo(OutputInfo* pOutInfo, 
//                                                  InputData* pData,
//                                                  BaseLearner* pWeakHypothesis);

        ////////////////////////////////////////////////
        // Internal structures
        ////////////////////////////////////////////////
        
        vector<BaseLearner*>  _foundHypotheses; //!< The list of the hypotheses found.
        
        string  _baseLearnerName; //!< The name of the basic learner used by AdaBoost. 
        string  _shypFileName; //!< File name of the strong hypothesis.
        string  _outputInfoFile; //!< The filename of the step-by-step information file that will be updated
        
        string  _trainFileName; //!< The filename of the input training set file
        string  _testFileName; //!< The filename of the input test set file
        
        int     _numIterations; //!< The number of iterations to run the algorithm
        
        /**
         * Verbose level.
         * There are three levels of verbosity:
         * - 0 = no messages
         * - 1 = basic messages
         * - 2 = show all messages
         */
        int     _verbose;
        const AlphaReal _smallVal; //!< A small value, to solve numeric issues
        
        bool _withConstantLearner; //!< Check or not constant learner in each iteration 
        
        string  _positiveLabelName; //!< The name of the face label in the input examples (provided in command line)
        int             _positiveLabelIndex; //!< The index of the face label (\see NameMap)
        
        ofstream _output; 
        int _sepWidth;  //!< The number of spaces that separate the columns in the output file
        
        string _trainPosteriorsFileName;    
        string _testPosteriorsFileName;    
        
        ////////////////////////////////////////////////
        // SoftCascade structures
        ////////////////////////////////////////////////
        
        double _targetDetectionRate; //!< The target true positives rate
        double _alphaExponentialParameter; //!< The parameter of the exponential distribution that shapres the rejection threshold vector
        
        string _unCalibratedShypFileName; //!< The input strong hypothesis to train on.
        
        bool _fullRun; //!< Indicates whether we produce first the uncalibrated strong hypothesis and then run the softcascade on the to top of it (not implemented)
        int _inShypLimit; //!< The number of weak hypotheses read from the input (uncalibrated) shyp file
        
        vector<AlphaReal> _rejectionThresholds; //!< The rejection found after a weak learner is selected (see the orginial paper)
        
        double _bootstrapRate; //!< The pourcentage of negative examples sampled at each iteration and added to the training set
        string _bootstrapFileName; //!< The name of the bootstrap file : a training data set containing the same attributes as the training set and only negatives.
        
        ////////////////////////////////////////////////////////////////
    private:
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        SoftCascadeLearner& operator=( const SoftCascadeLearner& ) {return *this;}
        
        /**
         * A temporary variable for h(x)*y. Helps saving time during re-weighting.
         */
        vector< vector<AlphaReal> > _hy;
        
    };
    
} // end of namespace MultiBoost

#endif // __SOFT_CASCADE_LEARNER_H
