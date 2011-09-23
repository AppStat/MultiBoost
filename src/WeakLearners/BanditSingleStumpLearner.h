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



#ifndef __BANDIT_SINGLE_STUMP_LEARNER_H
#define __BANDIT_SINGLE_STUMP_LEARNER_H

//#include "WeakLearners/ClasswiseLearner.h"
//#include "WeakLearners/FeaturewiseLearner.h"
#include "WeakLearners/SingleStumpLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"
#include "IO/SortedData.h"
#include "Utils/UCTutils.h"

#include "Bandits/GenericBanditAlgorithm.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
        
    enum BanditAlgo 
    {
        BA_RANDOM,
        BA_UCBK, // UCBK
        BA_UCBKV, // UCBKV
        BA_UCBKR, // UCBK randomzied
        BA_EXP3, // EXP3
        BA_EXP3G, // EXP3G
        BA_EXP3G2, // EXP3G
        BA_EXP3P // EXP3
    };
        
        
    /**
     * A \b single threshold decision stump learner using a bandit based feature selection policy!
     *  Here's the \b bibtex reference:
     \verbatim
     @InProceedings{BuKe10,
     author =       {Busa-Fekete, R. and K\'{e}gl, B.},
     title  =       {Fast boosting using adversarial bandits},
     booktitle =    {International Conference on Machine Learning},
     volume =          {27},
     year =         {2010},
     pages =        {143--150},
     location =     {Haifa, Israel}
     }
     }\endverbatim
     *
     */
    class BanditSingleStumpLearner : public SingleStumpLearner 
    {
    public:
        /**
         * The constructor.
         */
                
    BanditSingleStumpLearner() : SingleStumpLearner(), _banditAlgo( NULL ) {}
                
        /**
         * The destructor. Must be declared (virtual) for the proper destruction of 
         * the object.
         */
        virtual ~BanditSingleStumpLearner() {}
                
        /**
         */
        virtual void init();
                
                
        /**
         * Returns itself as object.
         * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
         * for the auto-registering classes.
         * \date 14/11/2005
         */
        virtual BaseLearner* subCreate() { 
            BaseLearner* retLearner = new BanditSingleStumpLearner();
            dynamic_cast< BanditSingleStumpLearner* >(retLearner)->setBanditAlgoObject( _banditAlgo );
            return retLearner;
        }
                
                
        /**
         * Declare weak-learner-specific arguments.
         * adding --baselearnertype
         * \param args The Args class reference which can be used to declare
         * additional arguments.
         * \date 24/04/2007
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
         * Run the learner to build the classifier on the given data.
         * \param pData The pointer to the data.
         * \see BaseLearner::run
         * \date 11/11/2005
         */
        virtual AlphaReal run();
                
        /**
         * Save the current object information needed for classification,
         * that is the single threshold.
         * \param outputStream The stream where the data will be saved
         * \param numTabs The number of tabs before the tag. Useful for indentation
         * \remark To fully save the object it is \b very \b important to call
         * also the super-class method.
         * \see StumpLearner::save()
         * \date 13/11/2005
         */
        virtual void save(ofstream& outputStream, int numTabs = 0);
                
        /**
         * Load the xml file that contains the serialized information
         * needed for the classification and that belongs to this class.
         * \param st The stream tokenizer that returns tags and values as tokens
         * \see save()
         * \date 13/11/2005
         */
        virtual void load(nor_utils::StreamTokenizer& st);
                
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
         * Returns a vector of float holding any data that the specific weak learner can generate
         * using the given input dataset. Right now just a single case is contemplated, therefore
         * "reason" is not used. The returned vector of data correspond to:
         * \f[
         * \left\{ v^{(t)}_1, v^{(t)}_2, \cdots, v^{(t)}_K, \phi^{(t)}({\bf x}_1), 
         * \phi^{(t)}({\bf x}_2), \cdots, \phi^{(t)}({\bf x}_n) \right\}
         * \f]
         * where \f$v\f$ is the alignment vector, \f$\phi\f$ is the discriminative function and
         * \f$t\f$ is the current iteration (to which this instantiation of weak learner belongs to).
         * \param data The vector with the returned data.
         * \param pData The pointer to the data points (the examples) previously loaded.
         * \remark This particular method has been created for analysis purposes (research). If you
         * need to get similar analytical information from this weak learner, add it to the function
         * and uncomment the parameter "reason".
         * \see BaseLearner::getStateData
         * \see Classifier::saveSingleStumpFeatureData
         * \date 10/2/2006
         * \remark TEMPORARLY OFF!!
         */
        //virtual void getStateData( vector<float>& data, const string& /*reason = ""*/, InputData* pData = 0 );
                
                
        /**
         * Calculate the reward from the edge according to the update rule
         * \param edge The edge value of the base learner.
         * \return reward value 
         * \date 10/11/2009
         */             
        AlphaReal getRewardFromEdge( AlphaReal edge );
                
                
        /**
         * Returns with the bandit object.
         * \return The bandit object.
         */
        virtual GenericBanditAlgorithm* getBanditAlgoObject() { return _banditAlgo; }

        /**
         * Returns with the bandit object.
         * \return The bandit object.
         */             
        virtual void setBanditAlgoObject( GenericBanditAlgorithm* banditAlgo ) { _banditAlgo = banditAlgo; }            
    protected:
        /*
          for EXP3G
        */
        void estimatePayoffs( vector<AlphaReal>& payoffs );
                
                
        // the notation is borrowed from the paper of Kocsis et. al. ECML
        //static vector< int > _T; // the number of a feature has been selected 
        //static int _numOfCalling; //number of the single stump learner have been called
        //static vector< float > _X; // the everage reward of a feature
        int _K; //<! The number of arms pulled at the same time.
        enum updateType _updateRule; //<! The way we calculate the reward based on the edge.
                
        AlphaReal       _reward;        //<! The reward we occurred.
                
        GenericBanditAlgorithm* _banditAlgo; //!< The pointer to the bandit object.
        BanditAlgo                              _banditAlgoName; //!< The name of the bandit algorithm.
                
        vector<AlphaReal>               _rewards;
        vector<int>                             _armsForPulling;
        AlphaReal                               _percentage; // for EXP3G
    };
        
    //////////////////////////////////////////////////////////////////////////
        
} // end of namespace MultiBoost

#endif // __SINGLE_STUMP_LEARNER_H

