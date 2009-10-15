#ifndef _Exp3G2_H
#define _Exp3G2_H

#include <list> 
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithm.h"
#include "Exp3G.h"
#include "Utils/Utils.h"

/*
The Exp3G2 Algorithm was published in: 
Kocsis and Szepesvari:
Reduced-Variance Payoff Estimation in Adversarial Bandit Problems
(it isn't available the journal where it was appeared, but it was written that it is published in the proceedings of ECML05)
*/


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {


class Exp3G2 : public Exp3G
{
public:
	Exp3G2(void);
	virtual ~Exp3G2(void) 
	{
	}

	virtual void receiveReward( int armNum, double reward );
	virtual void receiveReward( vector<double> reward );

	virtual void initialize( vector< double >& vals );

};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


} // end of namespace MultiBoost

#endif
