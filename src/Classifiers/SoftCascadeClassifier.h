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

//
//  SoftCascadeClassifier.h
//  MultiBoost
//

#ifndef __SOFT_CASCADE_CLASSIFIER_H
#define __SOFT_CASCADE_CLASSIFIER_H

//includes
#include <string>
#include "Utils/Args.h"
#include "Defaults.h"

using namespace std;

namespace MultiBoost {
    
    //forward declarations
    class InputData;
    class BaseLearner;
    class OutputInfo;
    
    class SoftCascadeClassifier {
        
    public:
        
        /**
         * The constructor. It initializes the variable and set them using the
         * information provided by the arguments passed. They are parsed
         * using the helpers provided by class Args.
         * \param args The arguments defined by the user in the command line.
         * \param verbose The level of verbosity
         * \see _verbose
         * \date 01/07/2011
         */
        SoftCascadeClassifier(const nor_utils::Args& args, int verbose = 1);
        
        
        /**
         * Starts the classification process. 
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * list of weak hypotheses that form the strong hypothesis.
         * \param outResFileName The name of the file in which the results of the classification
         * will be saved.
         * \param numRanksEnclosed This parameter defines the number of ranks to be printed.
         * \date 01/07/2011
         */
        void run(const string& dataFileName, const string& shypFileName, 
                 int numIterations, const string& outResFileName = "", 
                 int numRanksEnclosed = 2);

        /**
         * \param dataFileName The file name of the data to be classified.
         * \param shypFileName The strong hypothesis filename. It is the xml file containing the
         * list of weak hypotheses that form the strong hypothesis.
         * \param outResFileName The name of the file in which the results of the classification
         * will be saved.
         * \date 01/07/2011
         */
        void savePosteriors(const string& dataFileName, 
                            const string& shypFileName, 
                            const string& outFileName, 
                            int numIterations );
        
        
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
         * \date 01/07/2011
         */
        InputData* loadInputData(const string& dataFileName, const string& shypFileName);
        
        /**
         * Print interation-wise information.
         * \date 21/07/2011
         */
        void printOutputInfo(OutputInfo* pOutInfo, int t, 
                             InputData* pData, 
                             BaseLearner* pWeakHypothesis,
                             AlphaReal r);
        
        
        // internal structures
        /**
         * Defines the level of verbosity:
         * - 0 = no messages
         * - 1 = basic messages
         * - 2 = show all messages
         */
        int      _verbose;
        
        const nor_utils::Args&  _args;  //!< The arguments defined by the user.
        string   _outputInfoFile; //!< The filename of the step-by-step information file that will be updated 
        
        string  _positiveLabelName;

    private:
        
        /**
         * Fake assignment operator to avoid warning.
         * \date 6/12/2005
         */
        SoftCascadeClassifier& operator=( const SoftCascadeClassifier& ) {return *this;}
        

    };

} // end of namespace MultiBoost

#endif // __SOFT_CASCADE_CLASSIFIER_H
