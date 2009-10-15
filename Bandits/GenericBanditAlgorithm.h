#ifndef _GENERICBANDITALGORITHM_H
#define _GENERICBANDITALGORITHM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include "Utils/Utils.h"
#include "Utils/Args.h"
#include "Utils/StreamTokenizer.h"
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* An abstract class for a generic bandit algorithms.
* \see UCBK
 *\see Exp3
* \date 09/10/2009
*/


class GenericBanditAlgorithm
{
protected:
	int					_numOfArms;			// numer of the arms
	int					_numOfIter;			//number of the single stump learner have been called
	vector< int >		_T;					// the number of an arm has been selected 
	vector< double >	_X;					// the sum of reward for the features
	bool				_isInitialized;     // flag noting whter the object is initialized or not;

	bool				_serializationFlag;
	ofstream			_rewardFile;
public:
	GenericBanditAlgorithm(void) : _numOfArms( -1 ), _numOfIter( 2 ), _isInitialized( false ), _serializationFlag( false ) {}
	
	//initialize X and T vector
	void setArmNumber( int numOfArms )
	{
		//we can set the number of arm only once
		if ( _numOfArms < 0 ) 
		{		
			_numOfArms = numOfArms;
			_T.resize( _numOfArms );
			_X.resize( _numOfArms );		
		}

		fill( _T.begin(), _T.end(), 0 );
		fill( _X.begin(), _X.end(), 0.0 );
	}

	virtual int getArmNumber() { return _numOfArms; }

	
	//receive reward of only one arm
	virtual void receiveReward( int armNum, double reward )
	{
		if ( _serializationFlag )	
			writeOutActionAndReward( armNum, reward );

		_T[ armNum ]++;
		_X[ armNum ] += reward;
		incIter();
		updateithValue( armNum );		
	}

	virtual void incIter() { _numOfIter++; }
	virtual int getIterNum() { return _numOfIter; }

	
	virtual void displayArmStatistic( void )
	{
		for( int i=0; i < (int) _T.size(); i++ )
		{
			cout << i << ": " << _T[i] << " " << _X[i] << endl;
		}
	}

	
	virtual void getKBestAction( const int k, vector<int>& bestArms )
	{
		set<int> s;
		int i = 0;
		while ( i < _numOfArms )
		{
			s.insert( getNextAction() );
			if ( ((int)s.size())>=k) break;
			i++;
		}

		bestArms.clear();
		for( set<int>::iterator it = s.begin(); it != s.end(); it++ )
		{
			bestArms.push_back( *it );
		}	
	}

	//abstract functions
	virtual int getNextAction() = 0;

	// some getter and setter
	void setInitializedFlagToTrue() { _isInitialized = true; }
	bool isInitialized( void ) { return _isInitialized; }

	//initizlaize the init values of the arms 
	virtual void initialize( vector< double >& vals )
	{
		//for serialization
		if ( _serializationFlag ) writeOutInitialArray( vals );
		setInitializedFlagToTrue();
	}

	//serializetion
	void serializeInit( string fname )
	{
		_rewardFile.open( fname.c_str(), ofstream::binary );
		if ( _rewardFile.is_open() )
		{
			_serializationFlag = true;
		}
	}
	void serializeClose( void ) 
	{
		_rewardFile.close();
	}

	void writeOutActionAndReward( int armNum, double reward )
	{
		_rewardFile << armNum << " " << reward << endl;
	}

	void serializationLoad( string fname )
	{
		string line;
		ifstream in;

		in.open( fname.c_str(), ifstream::binary );

		if ( ! in.is_open() )
		{
			cout << "Bandit file doesn't exist!!" << endl;
			return;
		}

		size_t rowNum = nor_utils::count_rows( in );
		double reward;
		int armNum;
		string tmpVal;

		getline( in, line );
		
		vector< double > vals( 0 );

		istringstream ss( line );
		nor_utils::StreamTokenizer st(ss);

		while( st.has_token() )
		{
			tmpVal = st.next_token();
			stringstream ss( tmpVal );
			ss >> reward;
			vals.push_back( reward );
		}


		setArmNumber( (int)vals.size() );
		initialize( vals );

		for( size_t i = 1; i < rowNum; i++ ) 
		{
			getline( in, line );
			stringstream ssiter( line );
			ssiter >> armNum;
			ssiter >> reward;
			receiveReward( armNum, reward );
		}

		in.close();
		in.clear();

		_rewardFile.clear();
		_rewardFile.open( fname.c_str(), ofstream::app );
		if ( _rewardFile.is_open() )
		{
			_serializationFlag = true;
		}

	}

	virtual void initLearningOptions(const nor_utils::Args& args) = 0;
   /**
   * Declare arguments that belongs to all bandit algorithms. 
   * \remarks This method belongs only to this base class and must not
   * be extended.
   * \remarks I cannot use the standard declareArguments method, as it
   * is called only to instantiated objects, and as this class is abstract
   * I cannot do it.
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 10/2/2006
   */
   static void declareBaseArguments(nor_utils::Args& args)
	{
		args.setGroup("Bandit Algorithm Options");
		args.declareArgument("gamma", 
			"Exploation parameter.", 
			1, "<gamma>");   
		args.declareArgument("eta", 
			"Second parameter for EXP3G, EXP3.P", 
			1, "<eta>");
	}


protected:
	virtual void writeOutInitialArray( vector<double>& vals )
	{
		for( int i=0; i < (int)vals.size() - 1; i++ ) _rewardFile << vals[i] << " ";
		_rewardFile << vals[ vals.size()-1] << endl;
	}

	virtual void updateithValue( int armNum ) = 0;		
};

} // end of namespace MultiBoost

#endif