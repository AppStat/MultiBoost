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
 * \file AdaBoostMHLearner.h The meta-learner AdaBoostLearner.MH.
 */
#pragma warning( disable : 4786 )

#ifndef __ADABOOST_MH_LEARNER_H
#define __ADABOOST_MH_LEARNER_H

#include "StrongLearners/GenericStrongLearner.h"
#include "Utils/Args.h"
#include "Defaults.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
    
    class OutputInfo;
    class BaseLearner;
    class InputData;
    class Serialization;
    
    /**
     * The AdaBoost learner. This class performs the meta-learning
     * by calling the weak learners and updating the weights.
     * \date 12/11/2005
     */
    class AdaBoostMHLearner : public GenericStrongLearner
    {
    public:
        
        /**
         * The constructor. It initializes the variables and sets them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \date 13/11/2005
         */
        AdaBoostMHLearner()
        : _numIterations(0), _maxTime(-1), _theta(0), _verbose(1), _smallVal(1E-10),
        _resumeShypFileName(""), _outputInfoFile(""), _weightFile(""), _withConstantLearner(false), _fastResumeProcess(true) {}
        
        /**
         * Start the learning process.
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \see OutputInfo
         * \date 10/11/2005
         */
        virtual void run(const nor_utils::Args& args);

        /**
         * For SoftCascade
         * \param args The arguments provided by the command line with all
         * the options for training.
         * \see OutputInfo
         * \date 10/11/2005
         */        
		void run( const nor_utils::Args& args, InputData* pTrainingData, const string baseLearnerName, const int numIterations, vector<BaseLearner*>& foundHypotheses );
		
		
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
         * Updates the weights of the examples.
         * The re-weighting of \f$w\f$ (the weight vector over all the examples and classes)
         * is done using the following formula
         * \f[
         *  w^{(t+1)}_{i, \ell}=
         *        \frac{ w^{(t)}_{i, \ell} \exp \left( -\alpha^{(t)} 
         *        h_\ell^{(t)}(x_i) y_{i, \ell} \right) }{ Z^{(t)} }
         * \f]
         * where \a Z is a normalization factor, and it is defined as
         * \f[
         *  Z^{(t)} = 
         *     \sum_{j=1}^n \sum_{\ell=1}^k w^{(t)}_{j, \ell} \exp \left( -\alpha^{(t)} 
         *        h_\ell^{(t)}(x_j) y_{j, \ell} \right) 
         * \f]
         * where \f$n\f$ is the number of examples, \f$k\f$ the number of classes,
         * \f$\alpha\f$ is the confidence in the weak classifier,
         * \f$h_\ell(x_i)\f$ is the classification of example \f$x_i\f$ for class \f$\ell\f$ 
         * with the classifier found at the current iteration (see BaseLearner::classify()), 
         * and \f$y_i\f$ is the binary label of that 
         * example, defined in InputData::getBinaryClass().
         * \param pTrainingData The pointer to the training data.
         * \param pWeakHypothesis The current weak hypothesis.
         * \return The value of the edge. It will be used to see if the algorithm can continue
         * with learning.
         * \date 16/11/2005
         */
        AlphaReal updateWeights(InputData* pTrainingData, BaseLearner* pWeakHypothesis);
        
        /**
         * Updates the weights of the examples. If the slowresumeprocess is on, we do not calculate the 
         * weights in every iteration, but we calculate the re-weighting based on the exponential margins. 
         * Let's assume that it is given a strong learner \f$\f^{(t)}(\bx)\$f after t iteration.
         * Then the weights of the \f$t+1\f$ iteration can be written as
         * \f[
         * w^{(t+1)}_{i, \ell}=  
         *        \frac{ \exp \left( -f\ell^{(t)}(x_i) y_{i, \ell}\right) }
         *         { \sum_{i^{\prime}=1}^{N} \exp \left( -f\ell^{(t)}(x_{i^{\prime}}) y_{{i^{\prime}}, \ell}\right) }
         * \f]
         * Using this formula we avoid the iterationwise normalisation. But in this way we are not able to 
         * calulculate the the error rates in every iteration.
         */
        AlphaReal updateWeights(OutputInfo* pOutInfo, InputData* pData, vector<BaseLearner*>& pWeakHypothesis); //for fast resume process, because we calculate the weights of the samples based on the margin
        
        /**
         * Print output information if option --outputinfo is specified.
         * Called from run and resumeProcess
         * \see resumeProcess
         * \see run
         * \date 21/04/2007
         */
        void printOutputInfo(OutputInfo* pOutInfo, int t, InputData* pTrainingData, 
                             InputData* pTestData, BaseLearner* pWeakHypothesis);
        
        /**
         * Print out the weights of the instances at the end of the learning process if output file is given.
         * \see run
         * \date 29/04/2010
         */
        void printOutWeights( InputData* pData );
        
    protected:
        
        /**
         * Get the needed parameters (for the strong learner) from the argumens.
         * \param The arguments provided by the command line.
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
         * Resume the training using the features in _resumeShypFileName if the
         * option -resume has been specified.
         * \date 21/12/2005
         */
        void resumeProcess(Serialization& ss, InputData* pTrainingData, InputData* pTestData, 
                           OutputInfo* pOutInfo);
        
        vector<BaseLearner*>  _foundHypotheses; //!< The list of the hypotheses found.
        
        string  _baseLearnerName; //!< The name of the basic learner used by AdaBoost. 
        string  _shypFileName; //!< File name of the strong hypothesis.
        bool	   _isShypCompressed; 
        
        string  _trainFileName;
        string  _testFileName;
        
        int     _numIterations;
        int     _maxTime; //!< Time limit for the whole processing. Default: no time limit (-1).
        AlphaReal  _theta; //!< the value of theta. Default = 0.
        
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
        string  _weightFile; // !< The filename of the weights, if this is empty then it won't output the weights
        
        bool _withConstantLearner; //!< Check or not constant learner in each iteration 
        bool _fastResumeProcess; //Fast resume process (true), it will calculate only the error rate of the last iteration.
        ////////////////////////////////////////////////////////////////
    private:
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        AdaBoostMHLearner& operator=( const AdaBoostMHLearner& ) {return *this;}
        
        /**
         * A temporary variable for h(x)*y. Helps saving time during re-weighting.
         */
        vector< vector<AlphaReal> > _hy;
        
    };
    
} // end of namespace MultiBoost

#endif // __ADABOOST_MH_LEARNER_H
