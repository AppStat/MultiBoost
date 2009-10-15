#ifndef _RANDOM_H
#define _RANDOM_H

#include <list> 
#include <functional>
#include <math.h> //for log
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"
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

class Random : public GenericBanditAlgorithm
{
protected:
public:
	Random(void);
	virtual ~Random(void) {}

	virtual void getKBestAction( const int k, vector<int>& bestArms );
	virtual int getNextAction();

	virtual void initLearningOptions(const nor_utils::Args& args) {}
protected:
	virtual void updateithValue( int i );	
};

} // end of namespace MultiBoost

#endif
