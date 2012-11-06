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
 * \file TreeLearner.h Hamming tree, see the documentation.
 * \date 24/04/2007
 */

#ifndef __TREE_LEARNER_H
#define __TREE_LEARNER_H

#include "WeakLearners/BaseLearner.h"
#include "WeakLearners/ScalarLearner.h"

#include "Utils/Args.h"
#include "IO/InputData.h"

#include <vector>
#include <fstream>
#include <string>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	//////////////////////////////////////////////////////////////////////////
	
	template<class T>
	struct greater_first_tree : std::binary_function<T,T,bool>
	{
		inline bool operator()(const T& lhs, const T& rhs)
		{
			return lhs._edgeImprovement < rhs._edgeImprovement;
		}
	};
	
	struct NodePoint {
		ScalarLearner*	_learner;
		ScalarLearner*	_constantLearner;
		set< int >		_learnerIdxSet;
		AlphaReal		_edge;
		AlphaReal		_edgeImprovement;
		AlphaReal		_constantEdge;		
		
		AlphaReal		_constantEnergy;
		AlphaReal		_learnerEnergy;
		
		int			_idx;
		int			_parentIdx;
		int			_leftOrRightChild;
		bool		_extended;
		size_t		_size;
	};
	
	/**
	 * A learner that loads a set of base learners, and boosts on the top of them. 
	 */
	class TreeLearner : public virtual BaseLearner
	{
	public:
		
		
		/**
		 * The constructor. It initializes _numBaseLearners to -1
		 * \date 26/05/2007
		 */
		TreeLearner() : _numBaseLearners(-1), _pScalaWeakHypothesisSource( NULL ) { }
		
		/**
		 * The destructor. Must be declared (virtual) for the proper destruction of 
		 * the object.
		 */
		virtual ~TreeLearner() {
			for( int ib = 0; ib < _baseLearners.size(); ++ib )
				delete _baseLearners[ib];
		}
		
		/**
		 * Creates an InputData object using the base learner's createInputData.
		 * \see InputData
		 * \date 21/11/2005
		 */
		virtual InputData* createInputData() { return _pScalaWeakHypothesisSource->createInputData(); }
		
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
		 * Returns itself as object.
		 * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
		 * for the auto-registering classes.
		 * \date 14/11/2005
		 */
		virtual BaseLearner* subCreate() { return new TreeLearner(); }
		
		/**
		 * Run the learner to build the classifier on the given data.
		 * \see BaseLearner::run
		 * \date 24/04/2007
		 */
		virtual AlphaReal run();
		
		/**
		 * Return the classification using the learned classifier.
		 * \param pData The pointer to the data
		 * \param idx The index of the example to classify
		 * \param classIdx The index of the class
		 * \remark Passing the data and the index to the example is not nice at all.
		 * This will soon be replace with the passing of the example itself in some
		 * form (probably a structure to the example).
		 * \return the classification using the learned classifier.
		 * \date 24/04/2007
		 */
		virtual AlphaReal classify(InputData* pData, int idx, int classIdx);
		
		/**
		 * Save the current object information needed for classification,
		 * that is the single threshold.
		 * \param outputStream The stream where the data will be saved
		 * \param numTabs The number of tabs before the tag. Useful for indentation
		 * \remark To fully save the object it is \b very \b important to call
		 * also the super-class method.
		 * \see BaseLearner::save()
		 * \date 24/04/2007
		 */
		virtual void save(ofstream& outputStream, int numTabs = 0);
		
		/**
		 * Load the xml file that contains the serialized information
		 * needed for the classification and that belongs to this class.
		 * \param st The stream tokenizer that returns tags and values as tokens
		 * \see save()
		 * \date 24/04/2007
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
		 * \see TreeLearner::run()
		 * \date 25/05/2007
		 */
		virtual void subCopyState(BaseLearner *pBaseLearner);
		
		bool isBaseLearnerLeaf( int i ) const {
			return ( ( _idxPairs[i][0] == -1 ) && ( _idxPairs[i][1] == -1 ) );
		}
		
	protected:
		
		void extendNode( const NodePoint& parentNode, NodePoint& nodeLeft, NodePoint& nodeRight );
		void calculateEdgeImprovement( NodePoint& node );
		
		ScalarLearner* _pScalaWeakHypothesisSource;
		
		vector<ScalarLearner*> _baseLearners; //!< the learners of the product
		/*
		 If the _baseLearners[i] is classified as 1 then _idxPairs[i].first is the next classifier to be used, otherwise _idxPairs[i].second 
		 */
		vector< vector<int> > _idxPairs; //! The tree structure.
		int                   _numBaseLearners; //! Number of leaves in the tree structure.
	};
	
	
} // end of namespace MultiBoost

#endif // __TREE_LEARNER_H
