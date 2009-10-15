#ifndef _UCBK_RANDOMIZED_H
#define _UCBK_RANDOMIZED_H

#include <list> 
#include <functional>
#include <math.h> //for log
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"
#include "UCBK.h"
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
/*
class StorageElement{
protected:
	int		_index;
	double	_value;
public:
	int getIndex() { return _index; }
	double getValue() { return _value; }
	void setIndex( int index ) { _index = index; }
	void setValue( double value ) { _value = value; }
};
*/

class UCBKRandomized : public UCBK
{
public:
	UCBKRandomized(void);

	virtual void getKBestAction( const int k, vector<int>& bestArms );
	virtual int getNextAction();
protected:
	virtual void updateithValue( int i );
};

} // end of namespace MultiBoost

#endif
