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
 * \file VJCascadeClassifier.h Performs the classification with AdaBoostMH.
 */
#pragma warning( disable : 4786 )

#ifndef __VJCASCADE_CLASSIFIER_H
#define __VJCASCADE_CLASSIFIER_H

#include "Utils/Args.h"
#include "StrongLearners/VJCascadeLearner.h"

#include <string>
#include <cassert>

using namespace std;

namespace MultiBoost {
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    // Forward declarations.
    class ExampleResults;
    class InputData;
    class BaseLearner;
        
    /**
     * Classify a dataset with AdaBoost.MH learner.
     * Using the strong hypothesis file (shyp.xml by default) it builds the
     * list of weak hypothesis (or weak learners), and use them to perform a classification over 
     * the given data set. The strong hypothesis is the linear combination of the weak 
     * hypotheses and their confidence alpha, and is defined as:
     * \f[
     * {\bf g}(x) = \sum_{t=1}^T \alpha^{(t)} {\bf h}^{(t)}(x),
     * \f]
     * where the bold defines a vector as returned value.
     * To obtain a single class, we simply take the winning class that receives 
     * the "most vote", that is:
     * \f[
     * f(x) = \mathop{\rm arg\, max}_{\ell} g_\ell(x).
     * \f]
     * \date 15/11/2005
     */
    class VJCascadeClassifier
    {
    public:
                
        /**
         * The constructor. It initializes the variable and set them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \param args The arguments defined by the user in the command line.
         * \param verbose The level of verbosity
         * \see _verbose
         * \date 16/11/2005
         */
        VJCascadeClassifier(const nor_utils::Args& args, int verbose = 1);
                
        /**
         * Starts the classification process. 
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * list of weak hypotheses that form the strong hypothesis.
         * \param outResFileName The name of the file in which the results of the classification
         * will be saved.
         * \param numRanksEnclosed This parameter defines the number of ranks to be printed.
         * \remark If \a numRanksEnclosed=1, the only error displayed will be the one in which the
         * \f$\mathop{\rm arg\, max}_{\ell} g_\ell(x)\f$ is \b not the correct class.
         * \remark If \a outResFileName is provided, the result
         * of the classification will be saved in a file with the following format:
         * \verbatim
         1 className
         2 className
         3 className
         ...\endverbatim
         * If \a --examplelabel is active the first column of the data file will be used instead
         * of the number of example.
         * \date 16/11/2005
         */
        void run(const string& dataFileName, const string& shypFileName, 
                 int numIterations, const string& outResFileName = "" );
                
        /**
         * Print to stdout a nicely formatted confusion matrix.
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * list of weak hypotheses that form the strong hypothesis.
         * \date 10/2/2006
         */
        void printConfusionMatrix(const string& dataFileName, const string& shypFileName);
                
        /**
         * Output to a file a confusion matrix with every element separated by a tab.
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * list of weak hypotheses that form the strong hypothesis.
         * \param outFileName The name of the file in which the confusion matrix will be saved.
         * \param numIterations The number of weak learners to use
         * \date 10/2/2006
         */
        void saveConfusionMatrix(const string& dataFileName, const string& shypFileName,
                                 const string& outFileName);
                                                
                
        void savePosteriors(const string& dataFileName, const string& shypFileName,
                            const string& outFileName, int numIterations);
                
                
        /**
         * Ouptuts the forecast into  a file.
         */
        void outputForecast( InputData* pData, const string& outResFileName, vector<CascadeOutputInformation>& cascadeData );

        // for output
        virtual void outputHeader();
        virtual void outputCascadeResult( InputData* pData, vector<CascadeOutputInformation>& cascadeData );            
                
                
    protected:
                
        /**
         * Loads the data. It needs the Strong Hypothesis file because it needs
         * the information about the weak learner used to generate it. The weak
         * learner might have associated a special InputData derived class,
         * which is returned by BaseLearner::createInputData() once the weak
         * learner has been identified.
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * \warning The returned object must be destroyed by the caller.
         * \date 21/11/2005
         */
        InputData* loadInputData(const string& dataFileName, const string& shypFileName);
                                
        virtual void updateCascadeData(InputData* pData, vector<vector<BaseLearner*> >& weakHypotheses, 
                                       int stagei, const vector<AlphaReal>& posteriors, vector<AlphaReal>& thresholds, int positiveLabelIndex,
                                       vector<CascadeOutputInformation>& cascadeData);
        /**
         * Defines the level of verbosity:
         * - 0 = no messages
         * - 1 = basic messages
         * - 2 = show all messages
         */
        int      _verbose;
                
        const nor_utils::Args&  _args;  //!< The arguments defined by the user.
        string   _outputInfoFile; //!< The filename of the step-by-step information file that will be updated 
        string   _positiveLabelName;
        int              _positiveLabelIndex;
    private:
        ofstream _output; // for outputinfo
                
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        VJCascadeClassifier& operator=( const VJCascadeClassifier& ) {return *this;}
                
    };
        
} // end of namespace MultiBoost

#endif // __VJCASCADE_CLASSIFIER_H
