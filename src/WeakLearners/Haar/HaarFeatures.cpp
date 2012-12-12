
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


#include "HaarFeatures.h"
#include "IO/HaarData.h" // for areaWidth() and areaHeight()

#include <algorithm>

namespace MultiBoost {
        
    // ------------------------------------------------------------------------------
        
        
    HaarFeature::HaarFeature(short width, short height, const string& shortName, 
                             const string& name, eFeatureType type)
        : _shortName(shortName), _width(width), _height(height), _name(name), _type(type), 
          _accessType(AT_FULL) {}
        
    // ------------------------------------------------------------------------------
        
    void HaarFeature::fillHaarData( const vector<Example>& intImages, // in
                                    vector< pair<int, FeatureReal> >& haarData ) // out
    {
        switch (_type)
        {
        case FEATURE_2H_RECT: //!< Two horizontal.
            _fillHaarData<HaarFeature_2H>(intImages, haarData);
            break;
                                
        case FEATURE_2V_RECT: //!< Two vertical. 
                                
            _fillHaarData<HaarFeature_2V>(intImages, haarData);
            break;
                                
        case FEATURE_3H_RECT: //!< Three horizontal.
                                
            _fillHaarData<HaarFeature_3H>(intImages, haarData);
            break;
                                
        case FEATURE_3V_RECT: //!< Three vertical.
                                
            _fillHaarData<HaarFeature_3V>(intImages, haarData);
            break;
                                
        case FEATURE_4SQUARE_RECT: //!< Four square.
                                
            _fillHaarData<HaarFeature_4SQ>(intImages, haarData);
            break;
                                
        case FEATURE_NO_TYPE:
            cerr << "ERROR: fillHaarData called with no type\n";
            exit(-1);
            break;
        }
                
    }
        
    // ------------------------------------------------------------------------------
        
    int HaarFeature::precomputeConfigs()
    {
        _precomputedConfigs.clear();
                
        short x, y, w, h;
        size_t numConfigs = 0;
                
        // Get the number of configurations. I will replace this with a clean
        // formula soon.
        for (x = 0; x < HaarData::areaWidth(); ++x)
        {
            // Vertical pixel
            for (y = 0; y < HaarData::areaHeight(); ++y)
            {
                for (w = _width-1; w < HaarData::areaWidth() - x; w = w + _width)
                {
                    for (h = _height-1; h < HaarData::areaHeight() - y; h = h + _height)
                        ++numConfigs;
                } // w
            } // y
        } // x
                
        _precomputedConfigs.reserve(numConfigs);
                
        // Horizontal pixel
        for (x = 0; x < HaarData::areaWidth(); ++x)
        {
            // Vertical pixel
            for (y = 0; y < HaarData::areaHeight(); ++y)
            {
                for (w = _width-1; w < HaarData::areaWidth() - x; w = w + _width)
                {
                    for (h = _height-1; h < HaarData::areaHeight() - y; h = h + _height)
                        _precomputedConfigs.push_back( nor_utils::Rect(x, y, w, h) );
                } // w
            } // y
        } // x
                
        _configIt = _precomputedConfigs.begin();
        _visitedConfigs.resize(_precomputedConfigs.size());
        return static_cast<int>(_precomputedConfigs.size());
    }
        
    // ------------------------------------------------------------------------------
        
    inline 
    FeatureReal HaarFeature::getSumAt(const vector<FeatureReal>& intImage, int x, int y)
    {
#if MB_DEBUG
        if ( x >= HaarData::areaWidth() )
            cerr << "WARNING: x out of range: " << x << " >= " << HaarData::areaWidth() << endl;
        if ( y >= HaarData::areaHeight() )
            cerr << "WARNING: y out of range: " << y << " >= " << HaarData::areaHeight() << endl;
#endif 
        if (x < 0 || y < 0)
            return 0;
        else
            return intImage[ HaarData::areaWidth() * y + x ]; 
    }   
        
    // ------------------------------------------------------------------------------
        
    void HaarFeature::resetConfigIterator()
    {
        fill(_visitedConfigs.begin(), _visitedConfigs.end(), 0);
        _numVisited = 0;
        _configIt = _precomputedConfigs.begin();
        _loadedConfigIndex = 0;
                
        if (_accessType == AT_RANDOM_SAMPLING)
            this->moveToNextConfig();
    }
        
    // ------------------------------------------------------------------------------
    void HaarFeature::loadConfigByNum( int idx ) 
    {
        _loadedConfigIndex = idx;
        _configIt = _precomputedConfigs.begin() + idx; 
                
        assert(_configIt < _precomputedConfigs.end());
    }
        
    // ------------------------------------------------------------------------------
        
    void HaarFeature::moveToNextConfig()
    { 
        int num = 0;
        const FeatureReal confRange = static_cast<FeatureReal>(_precomputedConfigs.size()-1) / 
            static_cast<FeatureReal>(RAND_MAX);
                
        switch (_accessType)
        {
        case AT_RANDOM_SAMPLING:
                                
            // loop until a random number which has not been used yet has been found.
            // Note: this approach works very well when the number of possible configuration
            // is _very_ large.
            do {
                // pick up a random number btw 0 and the number of configurations
                num = static_cast<int>(static_cast<FeatureReal>( rand() ) * confRange);
                                        
            } while( _visitedConfigs[num] == 1 );
                                
            _loadedConfigIndex = num;
            _configIt = _precomputedConfigs.begin() + num; 
            _visitedConfigs[num] = 1;
            ++_numVisited;
                                
            assert(_configIt < _precomputedConfigs.end());
                                
            break;
                                
        case AT_FULL:
            ++_configIt;
            ++_loadedConfigIndex;
            break;
        }
    }
        
    // ------------------------------------------------------------------------------
        
    bool HaarFeature::hasConfigs() const
    { 
        switch(_accessType)
        {
        case AT_RANDOM_SAMPLING:
            if (_numVisited < _precomputedConfigs.size())
                return true;
            else
                return false;
            break;
                                
        case AT_FULL:
            return _configIt != _precomputedConfigs.end(); 
            break;
        }
                
        return false;
    }
        
    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------
        
    FeatureReal HaarFeature_2H::getValue(const vector<FeatureReal>& intImage, const nor_utils::Rect& r)
    {
        int xHalfPos = r.x + (r.width / 2);
        int yEndPos = r.y + r.height;
                
        FeatureReal whiteSum = getSumAt(intImage, xHalfPos, yEndPos) + // 4
            getSumAt(intImage, r.x-1, r.y-1) - // 1
            (getSumAt(intImage, xHalfPos, r.y-1) + // 2
             getSumAt(intImage, r.x-1, yEndPos) ); // 3
                
        xHalfPos++;
        FeatureReal blackSum = getSumAt(intImage, r.x+r.width, yEndPos) + // 4
            getSumAt(intImage, xHalfPos-1, r.y-1) - // 1
            (getSumAt(intImage, r.x+r.width, r.y-1) + // 2
             getSumAt(intImage, xHalfPos-1, yEndPos) ); // 3
                
        return blackSum - whiteSum;
    }
        
    // ------------------------------------------------------------------------------
        
        
    FeatureReal HaarFeature_2V::getValue(const vector<FeatureReal>& intImage, const nor_utils::Rect& r)
    {
        int yHalfPos = r.y + (r.height / 2);
        int xEndPos = r.x + r.width;
                
        FeatureReal whiteSum = getSumAt(intImage, xEndPos, yHalfPos) + // 4
            getSumAt(intImage, r.x-1, r.y-1) - // 1
            (getSumAt(intImage, xEndPos, r.y-1) + // 2
             getSumAt(intImage, r.x-1, yHalfPos) ); // 3
                
        yHalfPos++;
        FeatureReal blackSum = getSumAt(intImage, xEndPos, r.y+r.height) + // 4
            getSumAt(intImage, r.x-1, yHalfPos-1) - // 1
            (getSumAt(intImage, xEndPos, yHalfPos-1) + // 2
             getSumAt(intImage, r.x-1, r.y+r.height) ); // 3
                
        return blackSum - whiteSum;  
    }
        
    // ------------------------------------------------------------------------------
        
    FeatureReal HaarFeature_3H::getValue(const vector<FeatureReal>& intImage, const nor_utils::Rect& r)
    {
        // xOneThirdPos correspond to this (width = 8):
        //     x     A   B
        //    [0][1][2]|[3][4][5]|[6][7][8]
        // A = before whiteSum
        // B = after whiteSum
        int xOneThirdPos = r.x + ( r.width / 3);
                
        // twoThirdXPos correspond to this (width = 8):
        //     x               A   B
        //    [0][1][2]|[3][4][5]|[6][7][8]
        // A = before blackSum
        // B = after blackSum
        int  xTwoThirdPos = r.x + ( ((r.width+1) / 3) * 2 - 1);
                
        // Left White
        FeatureReal whiteSum = getSumAt(intImage, xOneThirdPos, r.y+r.height) + // 4
            getSumAt(intImage, r.x-1, r.y-1) - // 1
            (getSumAt(intImage, xOneThirdPos, r.y-1) + // 2
             getSumAt(intImage, r.x-1, r.y+r.height) ); // 3
                
        xOneThirdPos++;
                
        FeatureReal blackSum = getSumAt(intImage, xTwoThirdPos, r.y+r.height) + // 4
            getSumAt(intImage, xOneThirdPos-1, r.y-1) - // 1
            (getSumAt(intImage, xTwoThirdPos, r.y-1 ) + // 2
             getSumAt(intImage, xOneThirdPos-1, r.y+r.height) ); // 3
                
        xTwoThirdPos++;
                
        // Right White
        whiteSum += getSumAt(intImage, r.x+r.width, r.y+r.height) + // 4
            getSumAt(intImage, xTwoThirdPos-1, r.y-1) - // 1
            (getSumAt(intImage, r.x+r.width, r.y-1) + // 2
             getSumAt(intImage, xTwoThirdPos-1, r.y+r.height) ); // 3
                
        return blackSum - whiteSum;
    }
        
    // ------------------------------------------------------------------------------
        
    FeatureReal HaarFeature_3V::getValue(const vector<FeatureReal>& intImage, const nor_utils::Rect& r)
    {
        int yOneThirdPos = r.y + ( r.height / 3);
        int yTwoThirdPos = r.y + ( ((r.height+1) / 3) * 2 - 1);
                
        // Top White
        FeatureReal whiteSum = getSumAt(intImage, r.x+r.width, yOneThirdPos) + // 4
            getSumAt(intImage, r.x-1, r.y-1) - // 1
            (getSumAt(intImage, r.x+r.width, r.y-1) + // 2
             getSumAt(intImage, r.x-1, yOneThirdPos) ); // 3
                
        yOneThirdPos++;
                
        FeatureReal blackSum = getSumAt(intImage, r.x+r.width, yTwoThirdPos) + // 4
            getSumAt(intImage, r.x-1, yOneThirdPos - 1) - // 1
            (getSumAt(intImage, r.x+r.width, yOneThirdPos - 1 ) + // 2
             getSumAt(intImage, r.x-1, yTwoThirdPos) ); // 3
                
        yTwoThirdPos++;
                
        // Bottom White
        whiteSum += getSumAt(intImage, r.x+r.width, r.y+r.height) + // 4
            getSumAt(intImage, r.x-1, yTwoThirdPos-1) - // 1
            (getSumAt(intImage, r.x+r.width, yTwoThirdPos-1) + // 2
             getSumAt(intImage, r.x-1, r.y+r.height) ); // 3
                
        return blackSum - whiteSum;
    }
        
    // ------------------------------------------------------------------------------
        
    FeatureReal HaarFeature_4SQ::getValue(const vector<FeatureReal>& intImage, const nor_utils::Rect& r)
    {
        int yHalfPos = r.y + (r.height / 2);
        int yEndPos = r.y + r.height;
        int xHalfPos = r.x + (r.width / 2);
        int xEndPos = r.x + r.width;
                
        // Top left
        FeatureReal whiteSum = getSumAt(intImage, xHalfPos, yHalfPos) + // 4
            getSumAt(intImage, r.x-1, r.y-1) - // 1
            (getSumAt(intImage, xHalfPos, r.y-1) + // 2
             getSumAt(intImage, r.x-1, yHalfPos) ); // 3
                
        xHalfPos++;
                
        // Top right
        FeatureReal blackSum = getSumAt(intImage, xEndPos, yHalfPos) + // 4
            getSumAt(intImage, xHalfPos-1, r.y-1) - // 1
            (getSumAt(intImage, xEndPos, r.y-1) + // 2
             getSumAt(intImage, xHalfPos-1, yHalfPos) ); // 3
                
        xHalfPos--;
        yHalfPos++;
                
        // Adds bottom left
        blackSum += getSumAt(intImage, xHalfPos, yEndPos) + // 4
            getSumAt(intImage, r.x-1, yHalfPos-1) - // 1
            (getSumAt(intImage, xHalfPos, yHalfPos-1) + // 2
             getSumAt(intImage, r.x-1, yEndPos) ); // 3
                
        xHalfPos++;
                
        // Adds bottom right
        whiteSum += getSumAt(intImage, xEndPos, yEndPos) + // 4
            getSumAt(intImage, xHalfPos-1, yHalfPos-1) - // 1
            (getSumAt(intImage, xEndPos, yHalfPos-1) + // 2
             getSumAt(intImage, xHalfPos-1, yEndPos) ); // 3
                
        return blackSum - whiteSum;
    }
        
    // ------------------------------------------------------------------------------
        
} // end of namespace MultiBoost
