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


#include "TreeLearner.h"

#include "IO/Serialization.h"
#include "Others/Example.h"
#include "Utils/StreamTokenizer.h"
#include "Utils/Utils.h"

#include <cmath>
#include <limits>
#include <queue>
#include <typeinfo>

namespace MultiBoost {
        
    //REGISTER_LEARNER_NAME(Product, TreeLearner)

        
    // -----------------------------------------------------------------------
        
    void TreeLearner::declareArguments(nor_utils::Args& args)
    {
        BaseLearner::declareArguments(args);
                
        args.declareArgument("baselearnertype", 
                             "The name of the learner that serves as a basis for the tree\n"
                             "  and the number of base learners to be used in tree\n"
                             "  Don't forget to add its parameters\n",
                             2, "<baseLearnerType> <numBaseLearners>");
                
    }
        
    // ------------------------------------------------------------------------------
        
    void TreeLearner::initLearningOptions(const nor_utils::Args& args)
    {
        BaseLearner::initLearningOptions(args);
                
        string baseLearnerName;
        args.getValue("baselearnertype", 0, baseLearnerName);   
        args.getValue("baselearnertype", 1, _numBaseLearners);   
                
        // get the registered weak learner (type from name)
        BaseLearner* pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
                
        //check whether the weak learner is a ScalarLeaerner
        try {
            _pScalaWeakHypothesisSource = dynamic_cast<ScalarLearner*>(pWeakHypothesisSource);
        }
        catch (bad_cast& e) {
            cerr << "The weak hypothesis must be a ScalarLearner!!!" << endl;
            exit(-1);
        }
                
        _pScalaWeakHypothesisSource->initLearningOptions(args);
                
        /*
          for( int ib = 0; ib < _numBaseLearners; ++ib ) {                       
          vector< int > tmpVector( 2, -1 );
          _idxPairs.push_back( tmpVector );
          }
        */
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal TreeLearner::classify(InputData* pData, int idx, int classIdx)
    {               
        int ib = 0;
        while ( 1 ) {
            AlphaReal phix = _baseLearners[ib]->cut(pData,idx);
            if ( phix > 0 ) {
                if ( _idxPairs[ ib ][ 0 ] > 0 ) { // step down
                    ib = _idxPairs[ ib ][ 0 ];
                } else {
                    return _baseLearners[ib]->classify( pData, idx, classIdx ); 
                }
            } else if ( phix < 0 ) { 
                if ( _idxPairs[ ib ][ 1 ] > 0 ) { // step down
                    ib = _idxPairs[ ib ][ 1 ];
                } else {
                    return _baseLearners[ib]->classify( pData, idx, classIdx ); 
                }
            } else {
                return 0;
            }
                        
        }
    }
        
    // ------------------------------------------------------------------------------
        
    AlphaReal TreeLearner::run()
    {               
        set< int > tmpIdx, idxPos, idxNeg, origIdx;
        //ScalarLearner* pCurrentBaseLearner = 0;
        ScalarLearner* pTmpBaseLearner = 0;             
        int ib = 0;
        vector< int > tmpVector( 2, -1 );
                
        _pTrainingData->getIndexSet( origIdx );
                
                
        _pScalaWeakHypothesisSource->setTrainingData(_pTrainingData);
                
        //train the first learner               
        NodePoint parentNode, nodeLeft, nodeRight;                                                              
        parentNode._idx = 0;
        parentNode._parentIdx = -1;
        parentNode._learnerIdxSet = origIdx;
                
        calculateEdgeImprovement( parentNode );         
                
        // insert the root
        if ( parentNode._edgeImprovement < 0.0 ) // the constant is the best, in this case the treelearner is equivalent to the constant learner
        {
            _baseLearners.push_back( parentNode._constantLearner );                 
            _idxPairs.push_back( tmpVector );
            this->_alpha = parentNode._constantLearner->getAlpha();
            ib++;                   
            delete parentNode._learner;
            return parentNode._constantEnergy;
        }
                
        _baseLearners.push_back( parentNode._learner );
        _idxPairs.push_back( tmpVector );
        ib++;
                
        // put the first two children into the priority queue                                                           
        extendNode( parentNode, nodeLeft, nodeRight );
                
        calculateEdgeImprovement( nodeLeft );
        calculateEdgeImprovement( nodeRight );
                
        priority_queue< NodePoint, vector<NodePoint>, greater_first_tree<NodePoint> > pq;
                
        pq.push(nodeLeft);
        pq.push(nodeRight);
                
                
        while (ib < _numBaseLearners && ! pq.empty() )
        {
            NodePoint currentNode = pq.top();
            pq.pop();
                        
                        
            if ( _verbose > 3 ) {
                cout << "Current edge imporvement: " << currentNode._edgeImprovement << endl;
            }
                        
            if (currentNode._edgeImprovement>0)
            {
                _baseLearners.push_back( currentNode._learner );
                _idxPairs.push_back( tmpVector );
                //_baseLearners[ib] = currentNode._learner;
                delete currentNode._constantLearner;                            
            } else {
                _baseLearners.push_back(currentNode._constantLearner);
                _idxPairs.push_back( tmpVector );
                //_baseLearners[ib] = currentNode._constantLearner;
                delete currentNode._learner;            
                continue;
            }
                        
            _idxPairs[ currentNode._parentIdx ][ currentNode._leftOrRightChild ] = ib;
            currentNode._idx = ib;
            ib++;                                                                                                                                           
            if (ib >= _numBaseLearners) break;
                        
            extendNode( currentNode, nodeLeft, nodeRight );
                        
            calculateEdgeImprovement( nodeLeft );
            calculateEdgeImprovement( nodeRight );                                          
                        
            pq.push(nodeLeft);
            pq.push(nodeRight);                                             
        }
                
        while ( ! pq.empty() )
        {
            NodePoint currentNode = pq.top();
            pq.pop();
                        
            if (_verbose>3) cout << "Discarded node's edge improvement: " << currentNode._edgeImprovement << endl;
                        
            if (currentNode._learner) delete currentNode._learner;
            delete currentNode._constantLearner;
        }
                
        _id = _baseLearners[0]->getId();
        for(int ib = 1; ib < _baseLearners.size(); ++ib)
            _id += "_x_" + _baseLearners[ib]->getId();
                
        //calculate alpha
        this->_alpha = 0.0;
        AlphaReal eps_min = 0.0, eps_pls = 0.0;
                
        //_pTrainingData->clearIndexSet();
        _pTrainingData->loadIndexSet( origIdx );
        for( int i = 0; i < _pTrainingData->getNumExamples(); i++ ) {
            vector< Label> l = _pTrainingData->getLabels( i );
            for( vector< Label >::iterator it = l.begin(); it != l.end(); it++ ) {
                AlphaReal result  = this->classify( _pTrainingData, i, it->idx );
                                
                if ( ( result * it->y ) < 0 ) eps_min += it->weight;
                if ( ( result * it->y ) > 0 ) eps_pls += it->weight;
            }
                        
        }
                
        // set the smoothing value to avoid numerical problem
        // when theta=0.
        setSmoothingVal( (AlphaReal)(1.0 / _pTrainingData->getNumExamples() * 0.01 ) );
                
                
        this->_alpha = getAlpha( eps_min, eps_pls );
                
        // calculate the energy (sum of the energy of the leaves
        AlphaReal energy = this->getEnergy( eps_min, eps_pls );
                
        return energy;
    }
    // -----------------------------------------------------------------------
    void TreeLearner::extendNode( const NodePoint& parentNode, NodePoint& nodeLeft, NodePoint& nodeRight )
    {
        _pTrainingData->loadIndexSet( parentNode._learnerIdxSet );
                
        nodeLeft._learnerIdxSet.clear();
        nodeRight._learnerIdxSet.clear();
                
        //cut the dataset               
        for (int i = 0; i < _pTrainingData->getNumExamples(); ++i) {
            // this returns the phi value of classifier
            AlphaReal phix = parentNode._learner->cut(_pTrainingData,i);
            if ( phix <  0 )
                nodeLeft._learnerIdxSet.insert( _pTrainingData->getRawIndex( i ) );
            else if ( phix > 0 ) { // have to redo the multiplications, haven't been tested
                nodeRight._learnerIdxSet.insert( _pTrainingData->getRawIndex( i ) );
            }
        }
                
        nodeLeft._parentIdx = parentNode._idx;
        nodeRight._parentIdx = parentNode._idx;
                
        nodeLeft._leftOrRightChild = 1;
        nodeRight._leftOrRightChild = 0;                
                
        nodeLeft._extended = false;
        nodeRight._extended = false;    
                
        nodeLeft._learner = nodeLeft._constantLearner = NULL;
        nodeRight._learner = nodeRight._constantLearner = NULL;
                
        nodeLeft._size = nodeLeft._learnerIdxSet.size();
        nodeRight._size = nodeRight._learnerIdxSet.size();
    }
        
    // -----------------------------------------------------------------------
    void TreeLearner::calculateEdgeImprovement( NodePoint& node ) {
        node._extended = true;
        _pTrainingData->loadIndexSet( node._learnerIdxSet );
                
        // run constant
        BaseLearner* pConstantWeakHypothesisSource =
            BaseLearner::RegisteredLearners().getLearner("ConstantLearner");
                
        node._constantLearner = dynamic_cast<ScalarLearner*>( pConstantWeakHypothesisSource->create());
        node._constantLearner->setTrainingData(_pTrainingData);
        node._constantEnergy = node._constantLearner->run();
                
        node._constantEdge = node._constantLearner->getEdge(false);
        node._learner = NULL;
                
        if ( ! _pTrainingData->isSamplesFromOneClass() ) {
            node._learner = dynamic_cast<ScalarLearner*>(_pScalaWeakHypothesisSource->create());
            _pScalaWeakHypothesisSource->subCopyState(node._learner);
            node._learner->setTrainingData(_pTrainingData);
                        
            node._learnerEnergy = node._learner->run();
            if ( node._learnerEnergy == node._learnerEnergy ) { // isnan
                node._edge = node._learner->getEdge(false);
                node._edgeImprovement = node._edge - node._constantEdge;                                                                
            } else {
                node._edge = numeric_limits<AlphaReal>::signaling_NaN();
                node._edgeImprovement = -numeric_limits<AlphaReal>::max();
            }
        } else {
            node._edge = numeric_limits<AlphaReal>::signaling_NaN();
            node._edgeImprovement = 0.0;                    
        }
                
    }
        
        
    // -----------------------------------------------------------------------
        
    void TreeLearner::save(ofstream& outputStream, int numTabs)
    {
        // Calling the super-class method
        BaseLearner::save(outputStream, numTabs);
                
        // save numBaseLearners
        outputStream << Serialization::standardTag("numBaseLearners",  _baseLearners.size(), numTabs) << endl;
                
        for( int ib = 0; ib <  _baseLearners.size(); ++ib ) {
            outputStream << Serialization::standardTag("leftChild", _idxPairs[ib][0], numTabs) << endl;
            outputStream << Serialization::standardTag("rightChild", _idxPairs[ib][1], numTabs) << endl;
        }
                
        for( int ib = 0; ib < _baseLearners.size(); ++ib ) {
            //dynamic_cast<BaseLearner*>(_baseLearners[ib])->save(outputStream, numTabs + 1);
            _baseLearners[ib]->save(outputStream, numTabs + 1);
        }
    }
        
    // -----------------------------------------------------------------------
        
    void TreeLearner::load(nor_utils::StreamTokenizer& st)
    {
        BaseLearner::load(st);
                
        _numBaseLearners = UnSerialization::seekAndParseEnclosedValue<int>(st, "numBaseLearners");
        //   _numBaseLearners = 2;
        _idxPairs.clear();
        for(int ib = 0; ib < _numBaseLearners; ++ib) {
            int leftChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "leftChild");
            int rightChild = UnSerialization::seekAndParseEnclosedValue<int>(st, "rightChild");
            vector< int > p( 2, -1 );
            p[0] = leftChild;
            p[1] = rightChild;
            _idxPairs.push_back( p );
        }
                
        // kind of creepy solution. the vector of BaseLearner* cannot be casted to a vector of ScalarLearner*,
        // thus we shuold do this for each array element, separately.
        for(int ib = 0; ib < _numBaseLearners; ++ib) {
            vector<BaseLearner*> baseLearners(0);
            UnSerialization::loadHypothesis(st, baseLearners, _pTrainingData, _verbose);
                        
            if (baseLearners.size()>0)
            {
                _baseLearners.push_back( dynamic_cast<ScalarLearner*>(baseLearners[0]) );
                _baseLearners[ib]->cut(_pTrainingData,10);
            }
        }
                
    }
        
    // -----------------------------------------------------------------------
        
    void TreeLearner::subCopyState(BaseLearner *pBaseLearner)
    {
        BaseLearner::subCopyState(pBaseLearner);
                
        TreeLearner* pTreeLearner =
            dynamic_cast<TreeLearner*>(pBaseLearner);
                
        pTreeLearner->_numBaseLearners = _numBaseLearners;
                
        // deep copy
        for(int ib = 0; ib < _numBaseLearners; ++ib)
            pTreeLearner->_baseLearners.push_back(dynamic_cast<ScalarLearner*>(_baseLearners[ib]->copyState()));
    }
        
    // -----------------------------------------------------------------------
        
} // end of namespace MultiBoost
