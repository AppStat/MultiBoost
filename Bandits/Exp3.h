#ifndef _EXP3_H
#define _EXP3_H

#include <list> 
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"

using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

class Hedge : public GenericBanditAlgorithm {
protected:
	double _alpha;
	vector< double > _p;

public:
	
	Hedge( void ) : GenericBanditAlgorithm(), _p(0)
	{
		_alpha = 0.05;
	}

	//----------------------------------------------------------------
	//----------------------------------------------------------------
	// getters and setters 
	//----------------------------------------------------------------
	//----------------------------------------------------------------
	double getAlpha() { return _alpha; }
	void setAlpha( double alpha ) { _alpha = alpha; }

	void getProbabilityVector( vector< double >& p )
	{
		p.resize( _p.size() );
		copy( _p.begin(), _p.end(), p.begin() );
	}	

	//----------------------------------------------------------------
	// receive rewards for all arms!!!
	void receiveRewardVector( vector< double >& r )
	{
		for( int i=0; i < _numOfArms; i++ )
		{
			_T[i]++;
			_X[i] += r[i];
			incIter();
		}
		setPVector();
	}
	//----------------------------------------------------------------


	virtual void initialize( vector< double >& vals )
	{
		_p.resize( _numOfArms );
		setPVector();
	}

	virtual void getKBestAction( const int k, vector<int>& bestArms ) {}
	virtual int getNextAction() { return 0; }

	virtual void initLearningOptions(const nor_utils::Args& args) {}
protected:
	virtual void updateithValue( int armNum ) 
	{ 
		setPVector();
	}		

	void setPVector( void ) 
	{
		double sum = 0.0;

		for( int i=0; i < _numOfArms; i++ )
		{
			_p[i] = pow( ( 1 + _alpha ), _X[i] );
			sum += _p[i];
		}
		for( int i=0; i < _numOfArms; i++ )
		{
			_p[i] /= sum;
		}
	}
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


class Exp3 : public GenericBanditAlgorithm
{
protected:
	//double _alpha;
	double _gamma;
	vector< double > _p;
	vector< double > _pHat;

	//Hedge _hedge;
public:
	Exp3(void);
	virtual ~Exp3(void) 
	{
	}

	//----------------------------------------------------------------
	//----------------------------------------------------------------
	// getters and setters 
	//----------------------------------------------------------------
	//----------------------------------------------------------------
	//double getAlpha() { return _alpha; }
	//void setAlpha( double alpha ) { _alpha = alpha; }
	double getGamma() { return _gamma; }
	void setGamma( double gamma ) { _gamma = gamma; }

	virtual void receiveReward( int armNum, double reward );

	virtual void initialize( vector< double >& vals );

	virtual int getNextAction();

	virtual void initLearningOptions(const nor_utils::Args& args);
protected:
	virtual void updateithValue( int i );	
};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


} // end of namespace MultiBoost

#endif
