#ifndef _Exp3G_H
#define _Exp3G_H

#include <list> 
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"

/*
The Exp3G Algorithm was published in: 
Kocsis and Szepesvari:
Reduced-Variance Payoff Estimation in Adversarial Bandit Problems
(it isn't available the journal where it was appeared, but it was written that it is published in the proceedings of ECML05)
*/


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {


class Exp3G : public GenericBanditAlgorithm
{
protected:
	double _eta;
	double _gamma;
	vector< double > _p;
	vector< double > _w;
	vector< double > _tmpW;
	vector< vector< int > > _sideInformation;
	vector< int > _actions;
	//Hedge _hedge;
public:
	Exp3G(void);
	virtual ~Exp3G(void) 
	{
	}

	//----------------------------------------------------------------
	//----------------------------------------------------------------
	// getters and setters 
	//----------------------------------------------------------------
	//----------------------------------------------------------------
	double getEta() { return _eta; }
	void setEta( double eta ) { _eta = eta; }
	double getGamma() { return _gamma; }
	void setGamma( double gamma ) { _gamma = gamma; }

	virtual void receiveReward( int armNum, double reward );

	virtual void initialize( vector< double >& vals );
	virtual int getNextAction();

	virtual void initLearningOptions(const nor_utils::Args& args );
protected:
	virtual void updateithValue( int arm );	
};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


} // end of namespace MultiBoost

#endif
