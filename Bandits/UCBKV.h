#ifndef _UCBKVV_H
#define _UCBKVV_H

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

class UCBKV : public GenericBanditAlgorithm
{
protected:
   list< pair< double, int >* > _valuesList;
   vector< pair< double, int >* > _valueRecord;
   vector< vector< double > > _table;
   double _kszi; //exploaration paramerter
   double _c; //trade-off parameter
   double _b; // the support of rewards [0;b]

public:
	UCBKV(void);
	virtual ~UCBKV(void) 
	{
		list< pair< double, int >* >::iterator it;
		
		for( it = _valuesList.begin(); it != _valuesList.end(); it ++ )
		{
			delete *it;
		}

		_valuesList.clear();
		_valueRecord.clear();
	}
	
	virtual void receiveReward( int armNum, double reward );

	virtual void initialize( vector< double >& vals );

	virtual void getKBestAction( const int k, vector<int>& bestArms );
	virtual int getNextAction();

	virtual void initLearningOptions(const nor_utils::Args& args) {}
protected:
	virtual void updateithValue( int i );	
};

} // end of namespace MultiBoost

#endif
