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


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "Utils/Utils.h" // for addAndCheckExtension
#include "Defaults.h" // for defaultLearner
#include "IO/InputData.h"
#include "IO/Serialization.h" // to save the found strong hypothesis
#include "IO/OutputInfo.h"

#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/SoftCascadeLearner.h"
#include "Classifiers/SoftCascadeClassifier.h"
#include "StrongLearners/AdaBoostMHLearner.h"

namespace MultiBoost {

    // -----------------------------------------------------------------------------------

    void SoftCascadeLearner::getArgs(const nor_utils::Args& args)
    {
        if ( args.hasArgument("verbose") )
            args.getValue("verbose", 0, _verbose);

        ///////////////////////////////////////////////////
        // get the output strong hypothesis file name, if given
        if ( args.hasArgument("shypname") )
            args.getValue("shypname", 0, _shypFileName);
        else
            _shypFileName = string(SHYP_NAME);

        _shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);


        ///////////////////////////////////////////////////

        //TODO : create an abstract classe for cascade compliant base learners and accept only its offspring!
        // get the name of the learner
        _baseLearnerName = defaultLearner;
        if ( args.hasArgument("learnertype") )
            args.getValue("learnertype", 0, _baseLearnerName);
//            cout << "! Only HaarSingleStumpeLearner is allowed.\n";
        
        // -train <dataFile> <nInterations>
        if ( args.hasArgument("train") )
        {
            args.getValue("train", 0, _trainFileName);
            args.getValue("train", 1, _numIterations);
        }
        // -traintest <trainingDataFile> <testDataFile> <nInterations>
        else if ( args.hasArgument("traintest") ) 
        {
            args.getValue("traintest", 0, _trainFileName);
            args.getValue("traintest", 1, _testFileName);
            args.getValue("traintest", 2, _numIterations);
        }

        // The file with the step-by-step information
        if ( args.hasArgument("outputinfo") )
            args.getValue("outputinfo", 0, _outputInfoFile);
        
        
        // --constant: check constant learner in each iteration
        if ( args.hasArgument("constant") )
            _withConstantLearner = true;
        
        if ( args.hasArgument("positivelabel") )
        {
            args.getValue("positivelabel", 0, _positiveLabelName);
        } else {
            cout << "Error : The name of positive label must to given. \n Type --h softcascade to know the mandatory options." << endl;
            exit(-1);
        }
        
        if (args.hasArgument("trainposteriors")) {
            args.getValue("trainposteriors", 0, _trainPosteriorsFileName);
        }

        if (args.hasArgument("testposteriors")) {
            args.getValue("testposteriors", 0, _testPosteriorsFileName);
        }

        if (args.hasArgument("detectionrate")) {
            args.getValue("detectionrate", 0, _targetDetectionRate);
        }
        else {
            cout << "Error : the target detection rate must be given. \n Type --h softcascade to know the mandatory options.";
            exit(-1);
        }

        
        if (args.hasArgument("expalpha")) {
            args.getValue("expalpha", 0, _alphaExponentialParameter);
        }
        else {
            cout << "Error : the parameter used to initialize the rejection distribution vector must be given. \n Type --h softcascade to know the mandatory options.";
            exit(-1);
        }

        if (args.hasArgument("calibrate")) {
            args.getValue("calibrate", 0, _unCalibratedShypFileName);
            if (args.getNumValues("calibrate") > 1) {
                args.getValue("calibrate", 0, _inShypLimit);
            }
        }
        else {
            _fullRun = true;
            _unCalibratedShypFileName = "shypToBeCalibrated.xml";
            cout << "The strong hypothesis file will be seved into the file " << _unCalibratedShypFileName;
            //cout << "Error : the shyp file of the uncalibrated trained classifier must be given ! \n";
            //exit(-1);

        }
        
        if (args.hasArgument("bootstrap")) {
            cout << "Warning ! The bootstrapping set and the training set must come from the same superset. \n";
            args.getValue("bootstrap", 0, _bootstrapFileName);
            args.getValue("bootstrap", 1, _bootstrapRate);
        }
    }

    // -----------------------------------------------------------------------------------
    
    int SoftCascadeLearner::getInstanceLabel(InputData* pData, int i, int positiveLabelIndex) const
    {
        const Example& example = pData->getExample(i);
        int labelY = example.getLabelY(positiveLabelIndex);
            
        return (labelY+1) / 2; 
    }
    
    // -----------------------------------------------------------------------------------

    void SoftCascadeLearner::initializeRejectionDistributionVector(const int iNumberOfStages, vector<double>& oVector)
    {
        assert(iNumberOfStages > 0);
        
        double falseNegativesRate = 1 - _targetDetectionRate;
        
        oVector.resize(0);
        oVector.resize(iNumberOfStages, 0.);
        
        double vectorSum = 0.;
        
        if (_alphaExponentialParameter < 0)
        {
            for (int i = 0; i < iNumberOfStages; ++i) {
                oVector[i] = exp( - _alphaExponentialParameter * (1 - ((double)i / iNumberOfStages ) ) );
                vectorSum += oVector[i];
            }   
        }
        else 
        {
            for (int i = 0; i < iNumberOfStages; ++i) {
                oVector[i] = exp( _alphaExponentialParameter * (double(i) / iNumberOfStages ) );
                vectorSum += oVector[i];
            }
        }
        
        //normalization to satisfy the detection rate criteria
        double k = falseNegativesRate / vectorSum;
        for (int i = 0; i < iNumberOfStages; ++i) {
            oVector[i] *= k;
        }
    }

    // -----------------------------------------------------------------------------------
    
    AlphaReal SoftCascadeLearner::computeSeparationSpan(InputData* pData, const vector<AlphaReal> & iPosteriors, int positiveLabelIndex)
    {
        const int numExamples = pData->getNumExamples();
        const int numPositiveExamples = pData->getNumExamplesPerClass(_positiveLabelIndex);
        const int numNegativeExamples = pData->getNumExamplesPerClass(1 - _positiveLabelIndex);

        assert(numPositiveExamples > 0 && numNegativeExamples > 0);
        
        AlphaReal edgePos = 0., edgeNeg = 0.;

        for (int i = 0; i < numExamples; ++i) {

            AlphaReal posterior = 0. ;
            
            posterior = iPosteriors[i];
            
            edgePos += (posterior * getInstanceLabel(pData, i, positiveLabelIndex));
            edgeNeg += (posterior * ( 1  - getInstanceLabel(pData, i, positiveLabelIndex)));
        }
        
        return edgePos / numPositiveExamples - edgeNeg / numNegativeExamples;
    }
    
    // -----------------------------------------------------------------------------------
    
    void SoftCascadeLearner::updatePosteriors( InputData* pData, BaseLearner* weakHypotheses, vector<AlphaReal>& oPosteriors, int positiveLabelIndex )
    {
        const int numExamples = pData->getNumExamples();                
                
        AlphaReal alpha = weakHypotheses->getAlpha();

        for (int i = 0; i < numExamples; ++i)
        {
            oPosteriors[i] += alpha * weakHypotheses->classify(pData, i, positiveLabelIndex);
        }                       
    }
        
        
    // -------------------------------------------------------------------------
    
    void SoftCascadeLearner::computePosteriors(InputData* pData, vector<BaseLearner*> & weakHypotheses, vector<AlphaReal> & oPosteriors, int positiveLabelIndex)
    {
        const int numExamples = pData->getNumExamples();
        
        oPosteriors.resize(numExamples);
        fill(oPosteriors.begin(), oPosteriors.end(), 0. );
        
        vector<BaseLearner*>::iterator whyIt = weakHypotheses.begin();                          
        for (;whyIt != weakHypotheses.end(); ++whyIt )
        {
            BaseLearner* currWeakHyp = *whyIt;
            AlphaReal alpha = currWeakHyp->getAlpha();
                        
            for (int i = 0; i < numExamples; ++i)
            {
                AlphaReal alphaH = alpha * currWeakHyp->classify(pData, i, positiveLabelIndex);
                oPosteriors[i] += alphaH;
            }                       
        }
    }
    
    // -----------------------------------------------------------------------------------

  
    AlphaReal SoftCascadeLearner::findBestRejectionThreshold(InputData* pData, const vector<AlphaReal> & iPosteriors, const double & iFaceRejectionFraction, double & oMissesFraction)
    {
        const int numExamples = pData->getNumExamples();
        vector<pair<int, AlphaReal> > sortedPosteriors(iPosteriors.size());
        const int numPosExamples = pData->getNumExamplesPerClass(_positiveLabelIndex);

        double execptedNumDetections = numPosExamples * ( 1 - iFaceRejectionFraction );
        
        for (int i = 0; i < numExamples; ++i) {
            sortedPosteriors[i].first = getInstanceLabel(pData, i, _positiveLabelIndex);
            sortedPosteriors[i].second = iPosteriors[i];
        }
        
        sort( sortedPosteriors.begin(), sortedPosteriors.end(), nor_utils::comparePair<2, int, AlphaReal, greater<AlphaReal> >() );
                
        AlphaReal rejectionThesh = sortedPosteriors[0].second + numeric_limits<AlphaReal>::min();
        
        int detectedFaces = 0;
        int i;
        for (i = 0; i < numExamples; ++i) {
            
            if (sortedPosteriors[i].first == 1) {
                ++detectedFaces;
            }
            
            if ( (i != 0) && (sortedPosteriors[i].second != sortedPosteriors[i-1].second) ) {
                                
                rejectionThesh = (sortedPosteriors[i].second + sortedPosteriors[i-1].second)/2;
                if (detectedFaces > execptedNumDetections) {
                    break;
                }
            }
        }
        
        // could happen if we have only one whyp
        if (i == numExamples) {
            rejectionThesh = sortedPosteriors[numExamples - 1].second - 0.01;//numeric_limits<AlphaReal>::epsilon();
        }
        
        oMissesFraction = (double)(numPosExamples - detectedFaces) / numPosExamples;
        
        return rejectionThesh;
    }

    
    // -----------------------------------------------------------------------------------
    
    int SoftCascadeLearner::filterDataset(InputData* pData, const vector<AlphaReal> & posteriors, AlphaReal threshold, set<int> & indices)
    {
        const int numExamples = pData->getNumExamples();
        pData->getIndexSet(indices);
        
        int numExamplesRemoved = 0;
        
        for (int i = 0; i < numExamples; ++i) {
            int j = pData->getRawIndex(i);
            if (posteriors[i] < threshold) {
                indices.erase(j);
                ++numExamplesRemoved;
            }
        }
        
        pData->loadIndexSet( indices );
        int leftNegatives = pData->getNumExamplesPerClass(1-_positiveLabelIndex);
        cout << "[+] Dataset filtering :\t removed : " << numExamplesRemoved << endl;
        cout << "\t\t\t left negatives : " << leftNegatives << endl;
        return leftNegatives;
    }
    
    // -----------------------------------------------------------------------------------

    void SoftCascadeLearner::bootstrapTrainingSet(InputData * pData, InputData * pBootData, set<int> & indices)
    {
        set<int> bootIndices;
        pBootData->getIndexSet(bootIndices);
        
        const int numBootEx = pBootData->getNumExamples();
        const int K = (int)ceil(_bootstrapRate * numBootEx);
        
        cout << "[+] K = " << K << endl;
        
        int exampleCounter = 0;
//        for (int i = 0; i < numBootEx && exampleCounter < K ; ++i) {
        for (; exampleCounter < K ;) {            
            AlphaReal posterior = 0. ;
            bool forecasted = true;
            
            //TODO: assert the efficiency of the following
            int i = static_cast<int>(static_cast<FeatureReal>( rand() ) * (static_cast<FeatureReal>(K) - 1) / static_cast<FeatureReal>(RAND_MAX) );
            
            for (int s = 0; s < _foundHypotheses.size(); ++s) {                
                posterior += _foundHypotheses[s]->getAlpha() * _foundHypotheses[s]->classify(pData, i, _positiveLabelIndex);
                if ( posterior < _rejectionThresholds[s] ) {
                    forecasted = false;
                    break;
                }
            }
            
            if (forecasted) {
                assert(getInstanceLabel(pBootData, i, _positiveLabelIndex) == 0);
                ++exampleCounter;
                pData->addExample(pBootData->getExample(i));
                bootIndices.erase(pBootData->getRawIndex(i));
            }
        }
        
        pData->getIndexSet(indices);
        
        pBootData->loadIndexSet(bootIndices);
        
        //cout << "[+] number of bootstrapped examples : " << exampleCounter << endl;
        // no more bootstrapping
        if (exampleCounter == 0) {
            _bootstrapRate = 0;
        }
    }
    
#pragma mark -
    
    void SoftCascadeLearner::run(const nor_utils::Args& args)
    {
        // load the arguments
        this->getArgs(args);
        
        //print cascade properties
        if (_verbose > 0) {
            cout    << "[+] Softcascade parameters :" << endl
                    << "\t --> target detection rate = " << _targetDetectionRate << endl
                    << "\t --> alpha (exp param) = " << _alphaExponentialParameter << endl
                    << "\t --> bootstrap rate = " << _bootstrapRate << endl
                    << endl;
        }
        

        // get the registered weak learner (type from name)
        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
        // initialize learning options; normally it's done in the strong loop
        // also, here we do it for Product learners, so input data can be created
        pWeakHypothesisSource->initLearningOptions(args);

        // get the training input data, and load it

        InputData* pTrainingData = pWeakHypothesisSource->createInputData();
        pTrainingData->initOptions(args);
        pTrainingData->load(_trainFileName, IT_TRAIN, 5);

        InputData* pBootstrapData = NULL;
        if (!_bootstrapFileName.empty()) {
            pBootstrapData = pWeakHypothesisSource->createInputData();
            pBootstrapData->initOptions(args);
            pBootstrapData->load(_bootstrapFileName, IT_TRAIN, 5);
        }
        
        // get the testing input data, and load it
        InputData* pTestData = NULL;
        if ( !_testFileName.empty() )
        {
            pTestData = pWeakHypothesisSource->createInputData();
            pTestData->initOptions(args);
            pTestData->load(_testFileName, IT_TEST, 5);
        }

        Serialization ss(_shypFileName, false );
        ss.writeHeader(_baseLearnerName);
        
        
//        outputHeader();
        // The output information object
        OutputInfo* pOutInfo = NULL;

        if ( !_outputInfoFile.empty() ) 
        {
            pOutInfo = new OutputInfo(args, true);
            pOutInfo->setOutputList("sca", &args);
            
            pOutInfo->initialize(pTrainingData);
            
            if (pTestData)
                pOutInfo->initialize(pTestData);
            pOutInfo->outputHeader(pTrainingData->getClassMap(), true, true, false);
            pOutInfo->outputUserHeader("thresh");
            pOutInfo->headerEndLine();
        }
        
        
//        ofstream trainPosteriorsFile;
//        ofstream testPosteriorsFile;
        
        
        const NameMap& namemap = pTrainingData->getClassMap();
        _positiveLabelIndex = namemap.getIdxFromName(_positiveLabelName);

        // FIXME: output posteriors

//        OutputInfo* pTrainPosteriorsOut = NULL;
//        OutputInfo* pTestPosteriorsOut = NULL;
        
//        if (! _trainPosteriorsFileName.empty()) {
//            pTrainPosteriorsOut = new OutputInfo(_trainPosteriorsFileName, "pos", true);
//            pTrainPosteriorsOut->initialize(pTrainingData);
//            dynamic_cast<PosteriorsOutput*>( pTrainPosteriorsOut->getOutputInfoObject("pos") )->addClassIndex(_positiveLabelIndex );
//        }
        
//        if (! _testPosteriorsFileName.empty() && !_testFileName.empty() ) {
//            pTestPosteriorsOut = new OutputInfo(_testPosteriorsFileName, "pos", true);
//            pTestPosteriorsOut->initialize(pTestData);
//            dynamic_cast<PosteriorsOutput*>( pTestPosteriorsOut->getOutputInfoObject("pos") )->addClassIndex(_positiveLabelIndex );            
//        }
        
        const int numExamples = pTrainingData->getNumExamples();

        vector<BaseLearner*> inWeakHypotheses;
        
        if (_fullRun) {            
            // TODO : the full training is implementet, testing is needed
            AdaBoostMHLearner* sHypothesis = new AdaBoostMHLearner();
            sHypothesis->run(args, pTrainingData, _baseLearnerName, _numIterations, inWeakHypotheses );
            delete sHypothesis;
        }
        else { 
            
            cout << "[+] Loading uncalibrated shyp file... ";
            //read the shyp file of the trained classifier
            UnSerialization us;
            us.loadHypotheses(_unCalibratedShypFileName, inWeakHypotheses, pTrainingData);  
            if (_inShypLimit > 0 && _inShypLimit < inWeakHypotheses.size() ) {
                inWeakHypotheses.resize(_inShypLimit);
            }
            if (_numIterations > inWeakHypotheses.size()) {
                _numIterations = inWeakHypotheses.size();
            }
            cout << "weak hypotheses loaded, " << inWeakHypotheses.size() << " retained.\n";
        }
        
        // some initializations
        _foundHypotheses.resize(0);
        double faceRejectionFraction = 0.;
        double estimatedExecutionTime = 0.;
        vector<double> rejectionDistributionVector;

        _rejectionThresholds.resize(0);
        
        
        set<int> trainingIndices;
        for (int i = 0; i < numExamples; i++) {
            trainingIndices.insert(pTrainingData->getRawIndex(i) );
        }
        
        // init v_t (see the paper)
        initializeRejectionDistributionVector(_numIterations, rejectionDistributionVector);

        if (_verbose == 1)
            cout << "Learning in progress..." << endl;

        ///////////////////////////////////////////////////////////////////////
        // Starting the SoftCascade main loop
        ///////////////////////////////////////////////////////////////////////
        for (int t = 0; t < _numIterations; ++t)
        {
            if (_verbose > 0)
                cout << "--------------[ iteration " << (t+1) << " ]--------------" << endl;

            faceRejectionFraction += rejectionDistributionVector[t];
            
            cout << "[+] Face rejection tolerated : " << faceRejectionFraction << " | v[t] = " << rejectionDistributionVector[t] << endl;
            
            int numberOfNegatives = pTrainingData->getNumExamplesPerClass(1 - _positiveLabelIndex);
            
            //vector<BaseLearner*>::const_iterator whyIt;
            int selectedIndex = 0;
            AlphaReal bestGap = 0;
            vector<AlphaReal> posteriors;
            computePosteriors(pTrainingData, _foundHypotheses, posteriors, _positiveLabelIndex);
            
            //should use an iterator instead of i
            
            vector<BaseLearner*>::iterator whyIt;
            int i;
            for (i = 0, whyIt = inWeakHypotheses.begin(); whyIt != inWeakHypotheses.end(); ++whyIt, ++i) {
            
                vector<AlphaReal> temporaryPosteriors = posteriors;
                vector<BaseLearner*> temporaryWeakHyp = _foundHypotheses;
                temporaryWeakHyp.push_back(*whyIt);
                updatePosteriors(pTrainingData, *whyIt, temporaryPosteriors, _positiveLabelIndex);
                
                AlphaReal gap = computeSeparationSpan(pTrainingData, temporaryPosteriors, _positiveLabelIndex );

                if (gap > bestGap) {
                    bestGap = gap;
                    selectedIndex = i;
                }
            }
            
            BaseLearner* selectedWeakHypothesis = inWeakHypotheses[selectedIndex];
            
            cout << "[+] Rank of the selected weak hypothesis : " << selectedIndex << endl
                 << "\t ---> edge gap = " << bestGap << endl
                 << "\t ---> alpha = " << selectedWeakHypothesis->getAlpha() << endl;

            //update the stages
            _foundHypotheses.push_back(selectedWeakHypothesis);
            updatePosteriors(pTrainingData, selectedWeakHypothesis, posteriors, _positiveLabelIndex);
            
            double missesFraction;
            AlphaReal r = findBestRejectionThreshold(pTrainingData, posteriors, faceRejectionFraction, missesFraction);
            _rejectionThresholds.push_back(r);
            
            
            // update the output info object
            dynamic_cast<SoftCascadeOutput*>( pOutInfo->getOutputInfoObject("sca") )->appendRejectionThreshold(r);
            
            cout << "[+] Rejection threshold = " << r << endl;
            
            //some updates
            ss.appendHypothesisWithThreshold(t, selectedWeakHypothesis, r);
            faceRejectionFraction -= missesFraction;
            
            inWeakHypotheses.erase(inWeakHypotheses.begin() + selectedIndex);
            double whypCost = 1; //just in case there are different costs for each whyp
            estimatedExecutionTime += whypCost * numberOfNegatives;
            
            // output perf in file
            vector< vector< AlphaReal> > scores(0);
            _output << t + 1 << setw(_sepWidth + 1) << r << setw(_sepWidth);
            
            // update OutputInfo with the new whyp
//            updateOutputInfo(pOutInfo, pTrainingData, selectedWeakHypothesis);
//            if (pTestData) {
//                updateOutputInfo(pOutInfo, pTestData, selectedWeakHypothesis);
//            }
            

            // output the iteration results
            printOutputInfo(pOutInfo, t, pTrainingData, pTestData, selectedWeakHypothesis, r);
                        
//            if (pTrainPosteriorsOut) {
//                pTrainPosteriorsOut->setTable(pTrainingData, pOutInfo->getTable(pTrainingData));
//                pTrainPosteriorsOut->outputCustom(pTrainingData);
//            }
//
//            if (pTestPosteriorsOut) {
//                pTestPosteriorsOut->setTable(pTestData, pOutInfo->getTable(pTestData));
//                pTestPosteriorsOut->outputCustom(pTestData);
//            }
            
            
            int leftNegatives = filterDataset(pTrainingData, posteriors, r, trainingIndices);
            if (leftNegatives == 0) {
                cout << endl << "[+] No more negatives.\n";
                break;
            }
            
            if (_bootstrapRate != 0) {
                bootstrapTrainingSet(pTrainingData, pBootstrapData, trainingIndices);
            }

        }  // loop on iterations
        /////////////////////////////////////////////////////////

        // write the footer of the strong hypothesis file
        ss.writeFooter();

        // Free the two input data objects
        if (pTrainingData)
            delete pTrainingData;
        if (pBootstrapData) {
            delete pBootstrapData;
        }
        if (pTestData)
            delete pTestData;

        if (_verbose > 0)
            cout << "Learning completed." << endl;
    }
    
#pragma mark -

    // -------------------------------------------------------------------------
    
    void SoftCascadeLearner::printOutputInfo(OutputInfo* pOutInfo, int t, 
                                             InputData* pTrainingData, InputData* pTestData, 
                                             BaseLearner* pWeakHypothesis,
                                             AlphaReal r)
    {
        pOutInfo->outputIteration(t);
        pOutInfo->outputCustom(pTrainingData, pWeakHypothesis);

        if (pTestData)
        {
            pOutInfo->separator();
            pOutInfo->outputCustom(pTestData);    
        }

        pOutInfo->outputCurrentTime();
        pOutInfo->separator();
        pOutInfo->outputUserData(r);
        pOutInfo->endLine();

    }
    
    // -------------------------------------------------------------------------
    
//    void SoftCascadeLearner::updateOutputInfo(OutputInfo* pOutInfo, 
//                                              InputData* pData,
//                                              BaseLearner* pWeakHypothesis)
//    {
//        table& g = pOutInfo->getTable(pData);
//        
//        const long  numExamples = g.size();
//        
//        for (int j = 0; j < numExamples; ++j) {
//            
//            int i = pData->getRawIndex(j);
//            
//            vector<Label>& labels = pData->getLabels(i);
//            vector<Label>::const_iterator lIt;
//            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
//            {
//                // update the posteriors table
//                g[i][lIt->idx] += pWeakHypothesis->getAlpha() * // alpha
//                pWeakHypothesis->classify( pData, i, lIt->idx );             
//            }
//        }
//    }
    
    // -------------------------------------------------------------------------
    
    void SoftCascadeLearner::classify(const nor_utils::Args& args)
    {
        SoftCascadeClassifier classifier(args, _verbose);
                
        string testFileName = args.getValue<string>("test", 0);
        string shypFileName = args.getValue<string>("test", 1);
        int numIterations = args.getValue<int>("test", 2);
                
        string outResFileName = "";
        if ( args.getNumValues("test") > 3 )
            args.getValue("test", 3, outResFileName);
                
        classifier.run(testFileName, shypFileName, numIterations, outResFileName);
    }


    // -------------------------------------------------------------------------

    void SoftCascadeLearner::doPosteriors(const nor_utils::Args& args)
    {
        SoftCascadeClassifier classifier(args, _verbose);
        string testFileName = args.getValue<string>("posteriors", 0);
        string shypFileName = args.getValue<string>("posteriors", 1);
        string outFileName = args.getValue<string>("posteriors", 2);
        int numStages = args.getValue<int>("posteriors", 3);
                
        classifier.savePosteriors(testFileName, shypFileName, outFileName, numStages);

    }
        
    // -------------------------------------------------------------------------
    
    void SoftCascadeLearner::outputHeader()
    {
        _output.open(_outputInfoFile.c_str());
        
        if ( ! _output.is_open() ) {
            cout << "Cannot open output file" << endl;
            exit(-1);
        }
        
        _output << "--- new output ---" << endl;
        
        _output  << "it" << setw(_sepWidth)  << "th" << setw(_sepWidth) << "|" ;
        
        _output << setw(_sepWidth) << "err";
        _output << setw(_sepWidth) << "auc";
        _output << setw(_sepWidth) << "fpr";
        _output << setw(_sepWidth) << "tpr";
        _output << setw(_sepWidth) << "nbeval";
        
        _output << setw(_sepWidth) << "|"  << setw(_sepWidth) << "err";
        _output << setw(_sepWidth) << "auc";
        _output << setw(_sepWidth) << "fpr";
        _output << setw(_sepWidth) << "tpr";
        _output << setw(_sepWidth) << "nbeval";
        
        _output << endl ;
    }
    
    // -------------------------------------------------------------------------

    void SoftCascadeLearner::outputCascadePerf(InputData* pData, vector< vector< AlphaReal> > & outScores)
    {
        set<int> indices ;
        pData->getIndexSet(indices);
        pData->clearIndexSet();
        
        const int numExamples = pData->getNumExamples();
        
        _output << setprecision(4) << "|";
        
        int P = pData->getNumExamplesPerClass(_positiveLabelIndex);
        int N = pData->getNumExamplesPerClass(1 - _positiveLabelIndex);
        int TP = 0, FP = 0;
        int err = 0;
        int numWhyp = 0;
        
        //compute the posteriors
//        vector<AlphaReal> posteriors(numExamples, 0.); 
//        computePosteriors(pData, _foundHypotheses, posteriors, _positiveLabelIndex);
        
        //for the ROC curve
        vector< pair< int, AlphaReal> > scores;
        scores.resize(0);
        scores.resize(numExamples);
        outScores.resize(numExamples);
        
        AlphaReal alphaSum = 0. ;
        for (int wh = 0; wh < _foundHypotheses.size(); ++wh) {
            alphaSum += _foundHypotheses[wh]->getAlpha();
        }
        
        for (int i = 0; i < numExamples; ++i) {
            
            int forecast = 1;
            AlphaReal posterior = 0. ;
            
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);

            int nbEvaluations = 0 ;
            
            for (int s = 0; s < _foundHypotheses.size(); ++s) {
                
                nbEvaluations += 1;

                posterior += _foundHypotheses[s]->getAlpha() * _foundHypotheses[s]->classify(pData, i, _positiveLabelIndex);
                if ( posterior < _rejectionThresholds[s] ) {
                    forecast = -1;
                    break;
                }
            }

            scores[i].second = ( ( posterior/alphaSum ) + 1 ) / 2 ;                 

            if (labelY < 0) {
                numWhyp += nbEvaluations;
                scores[i].first = 0;
            }
            else 
                scores[i].first = 1;
            
            outScores[i].resize(2);
            outScores[i][0] = scores[i].second;
            outScores[i][1] = nbEvaluations;
            
            if (forecast * labelY < 0) {
                err++;
            }
            
            if (forecast > 0)
            {
                if (labelY > 0) {
                    TP++;
                }
                else {
                    FP++;
                }                
            }
        }
                
        double tpRate = (double)TP/P;
        double fpRate = (double)FP/N;
        double eval = (double)numWhyp / N;
        double errRate = (double)err/numExamples;
        
        double rocScore = nor_utils::getROC( scores );  
        
        _output << setw(_sepWidth) << errRate;
        _output << setw(_sepWidth) << rocScore;
        _output << setw(_sepWidth) << fpRate;
        _output << setw(_sepWidth) << tpRate;
        _output << setw(_sepWidth) << eval;
        
        pData->loadIndexSet(indices);
    }
    
    // -------------------------------------------------------------------------


} // end of namespace MultiBoost

