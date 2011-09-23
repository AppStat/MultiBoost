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


#include "IO/Serialization.h"
#include "Utils/Utils.h" // for cmp_nocase
#include "WeakLearners/BaseLearner.h"
#include "Bzip2/Bzip2Wrapper.h"

#include <cctype> // for isspace

namespace MultiBoost {
        
    // -----------------------------------------------------------------------
        
    Serialization::Serialization(const string& shypFileName, bool isComp )
    { 
        _shypFile.open(shypFileName.c_str());
        _shypFileName = shypFileName;
        _isComp = isComp;
        _bzipFileName = _shypFileName;
        _bzipFileName.append( ".bz2" );
        remove( _bzipFileName.c_str() );
    }
        
    // -----------------------------------------------------------------------
        
    void Serialization::writeHeader(const string& weakLearnerName)
    {
        // print the header
        _shypFile << "<?xml version=\"1.0\"?>" << endl;
        _shypFile << "<multiboost>" << endl;
        _shypFile << standardTag("algo", weakLearnerName, 1) << endl;
                
        // print general information relative to the weak learner
        //    BaseLearner::RegisteredLearners().getLearner(weakLearnerName)->saveGeneral(_shypFile, 1);
    }
        
    // -----------------------------------------------------------------------
        
    void Serialization::writeCascadeHeader(const string& weakLearnerName)
    {
        // print the header
        _shypFile << "<?xml version=\"1.0\"?>" << endl;
        _shypFile << "<cascade>" << endl;
        _shypFile << standardTag("algo", weakLearnerName, 1) << endl;
                
    }
        
        
    // -----------------------------------------------------------------------
        
    void Serialization::writeFooter()
    {
        // close tag
        _shypFile << "</multiboost>" << endl;
                
        if ( _isComp ) {
            flushCompressedBuffer();
        }
                
    }
    // -----------------------------------------------------------------------
        
    void Serialization::writeCascadeFooter()
    {
        // close tag
        _shypFile << "</cascade>" << endl;
                
        if ( _isComp ) {
            flushCompressedBuffer();
        }
                
    }
        
        
    // -----------------------------------------------------------------------
    void Serialization::flushCompressedBuffer()
    {
        _shypFile.flush();
        //cout << "Shypfile size:\t" << _shypFile.tellp() << endl;
                
        _shypFile.close();
        _shypFile.clear();
                
        ifstream inFile;
        string str;
        char buf[ 4086 ];
        inFile.open( _shypFileName.c_str() );
                
        Bzip2WrapperWriter bzw;
                
        bzw.open( _bzipFileName.c_str(), true );
                
        getline( inFile, str );
        while ( inFile ) {
            sprintf( buf, "%s\n", str.c_str() );
            bzw << buf;
            getline( inFile, str );
        }
                
                
        inFile.close();
        bzw.close();
        remove( _shypFileName.c_str() );
                
    }
    // -----------------------------------------------------------------------
        
    void Serialization::saveHypotheses(vector<BaseLearner*>& weakHypotheses)
    {
        // save the weak hypotheses one by one.
        for (int i = 0; i < (int)weakHypotheses.size(); ++i)
            appendHypothesis(i, weakHypotheses[i]);
    }
        
    // -----------------------------------------------------------------------      
    void Serialization::appendStageSeparatorHeader( int stageIndex, int weakhypnum, double threshold )
    {               
        //_shypFile << "\t<stage num=\""  << stageIndex << "\" weakhypnum=\"" <<  weakhypnum << "\" threshold=\""<< threshold << "\">" << endl;
        _shypFile << "\t<stage num=\""  << stageIndex << "\">" << endl; 
        _shypFile << "\t\t<weakhypnum>" <<  weakhypnum << "</weakhypnum>" << endl;
        _shypFile << "\t\t<threshold>"  << threshold << "</threshold>" << endl;
        _shypFile << "\t</stage>" << endl;
        _shypFile << "\t<!-- ################################## -->" << endl;
    }
        
    // -----------------------------------------------------------------------      
    /*
      void Serialization::appendStageSeparatorFooter()
      {
                
      }
    */
        
    // -----------------------------------------------------------------------
        
    void  Serialization::appendHypothesis(int iteration, BaseLearner* pWeakHypothesis, int numTab)
    {
        // just for readability
        string shiftTab = getTabs(numTab);
        
        // open the hypothesis tag (parameters: iteration, weak learner's name)
        _shypFile << shiftTab << "\t<weakhyp iter=\"" << iteration << "\">" << endl;
                
        // save the hypothesis
        pWeakHypothesis->save(_shypFile, 2 + numTab); 
                
        // close the hypothesis tag
        _shypFile << shiftTab << "\t</weakhyp>"<< endl;
                
        // add a separation "comment"
        _shypFile << "\t<!-- ################################## -->" << endl;
                
        if ( _isComp ) {
            _shypFile.flush();
            //cout << "Shypfile size:\t" << _shypFile.tellp() << endl;
                        
            if ( _shypFile.tellp() > 10e6 ) {
                _shypFile.close();
                _shypFile.clear();
                                
                ifstream inFile;
                string str;
                char buf[ 4086 ];
                inFile.open( _shypFileName.c_str() );
                                
                Bzip2WrapperWriter bzw;
                                
                bzw.open( _bzipFileName.c_str(), true );
                                
                getline( inFile, str );
                while ( inFile ) {
                    sprintf( buf, "%s\n", str.c_str() );
                    bzw << buf;
                    getline( inFile, str );
                }
                                
                                
                inFile.close();
                bzw.close();
                _shypFile.open( _shypFileName.c_str() );
            }
        }
    }
        
    // -----------------------------------------------------------------------

    void  Serialization::appendHypothesisWithThreshold(int iteration, BaseLearner* pWeakHypothesis, double threshold, int numTab)
    {
        // just for readability
        string shiftTab = getTabs(numTab);
        
        // open the hypothesis tag (parameters: iteration, weak learner's name)
        _shypFile << shiftTab << "\t<weakhyp iter=\"" << iteration << "\">" << endl;
        
        _shypFile << shiftTab << "\t\t<rejecthresh>" << threshold << "</rejecthresh>" << endl ;
                
        // save the hypothesis
        pWeakHypothesis->save(_shypFile, 2 + numTab); 
                
        // close the hypothesis tag
        _shypFile << shiftTab << "\t</weakhyp>"<< endl;
                
        // add a separation "comment"
        _shypFile << "\t<!-- ################################## -->" << endl;
                
        if ( _isComp ) {
            _shypFile.flush();
            //cout << "Shypfile size:\t" << _shypFile.tellp() << endl;
                        
            if ( _shypFile.tellp() > 10e6 ) {
                _shypFile.close();
                _shypFile.clear();
                                
                ifstream inFile;
                string str;
                char buf[ 4086 ];
                inFile.open( _shypFileName.c_str() );
                                
                Bzip2WrapperWriter bzw;
                                
                bzw.open( _bzipFileName.c_str(), true );
                                
                getline( inFile, str );
                while ( inFile ) {
                    sprintf( buf, "%s\n", str.c_str() );
                    bzw << buf;
                    getline( inFile, str );
                }
                                
                                
                inFile.close();
                bzw.close();
                _shypFile.open( _shypFileName.c_str() );
            }
        }
    }
        
    // -----------------------------------------------------------------------    
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
        
    void UnSerialization::loadHypotheses(const string& shypFileName, 
                                         vector<BaseLearner*>& weakHypotheses,
                                         InputData* pTrainingData, int verbose)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }
                
        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
                
        // Move until it finds the multiboost tag
        if ( !seekSimpleTag(st, "multiboost") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
            exit(1);
        }
                
        // Move until it finds the algo tag
        // Not used for the moment
        string algoName = seekAndParseEnclosedValue<string>(st, "algo");
                
        // Will be filled in by seekAndParseParamTag in each iteration
        for ( ; ; )
        {
            // move until the next weak hypothesis
            if ( seekParamTag(st, "weakhyp") )
                loadHypothesis(st, weakHypotheses, pTrainingData, verbose);
            else
                break;
        }
        cout << weakHypotheses.size() << endl;
                
    }

    void UnSerialization::loadHypothesesWithThresholds(const string& shypFileName, 
                                                       vector<BaseLearner*>& weakHypotheses,
                                                       vector<AlphaReal>& rejectionThresholds,
                                                       InputData* pTrainingData, int verbose)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }
                
        rejectionThresholds.resize(0);
        
        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
                
        // Move until it finds the multiboost tag
        if ( !seekSimpleTag(st, "multiboost") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
            exit(1);
        }
                
        // Move until it finds the algo tag
        // Not used for the moment
        string algoName = seekAndParseEnclosedValue<string>(st, "algo");
                
        // Will be filled in by seekAndParseParamTag in each iteration
        for ( ; ; )
        {   
            // move until the next weak hypothesis
            if ( seekParamTag(st, "weakhyp") ) {
                double threshold = seekAndParseEnclosedValue<double>(st, "rejecthresh");
                rejectionThresholds.push_back(threshold);
                loadHypothesis(st, weakHypotheses, pTrainingData, verbose);
            }
            else
                break;
            
        }
        cout << weakHypotheses.size() << endl;
                
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------      
    void UnSerialization::loadHypothesis(nor_utils::StreamTokenizer& st, 
                                         vector<BaseLearner*>& weakHypotheses,
                                         InputData* pTrainingData, int verbose)
    {
        string basicLearnerName =  seekAndParseEnclosedValue<string>(st, "weakLearner");
                
        // Check if the weak learner exists
        if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) ) {
            cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
            exit(1);
        }
                
        // allocate the weak learner object
        BaseLearner* pWeakHypothesis = 
            BaseLearner::RegisteredLearners().getLearner(basicLearnerName)->create();
        pWeakHypothesis->setTrainingData(pTrainingData);
                
        // load it
        pWeakHypothesis->load(st);
                
        // at least </weakhyp> should be expected,
        // therefore this was a broken weak learner
        if ( !st.has_token() ) {
            cerr << "WARNING: Incomplete weak hypothesis file found. Check the shyp file!" << endl;
            delete pWeakHypothesis;
            return;
        }
        // store it in the vector
        weakHypotheses.push_back(pWeakHypothesis);
                
        // show some progress while loading on verbose > 1
        if (verbose > 1 && weakHypotheses.size() % 1000 == 0)
            cout << "." << flush;
    }
        
    // -----------------------------------------------------------------------
        
    string UnSerialization::getWeakLearnerName(const string& shypFileName)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }
                
        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
                
        // Declares the possible starting tag names for stronglearners
        vector<string> tags;
        tags.push_back( "multiboost" );
        tags.push_back( "cascade" );
                
        // Move until it finds the multiboost tag
        //if ( !seekSimpleTag(st, "multiboost") )
        if ( !seekSimpleTag(st, tags) )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
            exit(1);
        }
                
        // Move until it finds the algo tag
        return seekAndParseEnclosedValue<string>(st, "algo");
    }

    // -----------------------------------------------------------------------
        
    bool UnSerialization::seekSimpleTag(nor_utils::StreamTokenizer& st, const vector<string>& tags)
    {
        do{
            string nextToken = st.next_token();
            vector<string>::const_iterator it = tags.begin();
            for( ; it != tags.end(); ++it) {
                if ( nor_utils::cmp_nocase( nextToken, *it ) )
                    return true;
            }
        } while( st.has_token() );
                
        return false;   
    }
        
        
    // -----------------------------------------------------------------------
        
    bool UnSerialization::seekSimpleTag(nor_utils::StreamTokenizer& st, const string& tag)
    {
        do{
            if ( nor_utils::cmp_nocase( st.next_token(), tag ) )
                return true;
        } while( st.has_token() );
                
        return false;   
    }
        
    // -----------------------------------------------------------------------
        
    bool UnSerialization::seekParamTag(nor_utils::StreamTokenizer& st, const string& tag)
    {
                
        do {
            // the full tag. I.e. <tag param="val">
            string rawTag = st.next_token();
            string tagOnly;
            string::const_iterator p = rawTag.begin();
                        
            // get tag name
            insert_iterator<string> tagIt(tagOnly, tagOnly.begin());
            for ( ; p != rawTag.end(); ++p)
            {
                if ( isspace(*p) )
                    break;
                *tagIt = *p;
            }
                        
            // check if it is the one we are looking for
            if ( nor_utils::cmp_nocase( tagOnly, tag ) )
                return true;
                        
        } while( st.has_token() );
                
        return false;
    }
        
    // -----------------------------------------------------------------------
        
    void UnSerialization::parseParamTag(const string& str, string& tag, string& tagParam, string& paramValue)
    {
        // simple tag. Return just the string
        if ( str.find('=') == string::npos )
        {
            tag = str;
            return;
        } 
                
        tag = "";
        tagParam = "";
        paramValue = "";
                
        string::const_iterator p = str.begin();
                
        // get tag name
        insert_iterator<string> tagIt(tag, tag.begin());
        for ( ; p != str.end(); ++p)
        {
            if ( isspace(*p) )
                break;
            *tagIt = *p;
        }
                
        // skip white spaces
        for ( ; isspace(*p) && p != str.end() ; ++p );
                
        // get param name
        insert_iterator<string> paramIt(tagParam, tagParam.begin());
        for ( ; p != str.end(); ++p)
        {
            if (*p == '=')
                break;
            *paramIt = *p;
        }
                
        // skip white spaces
        for ( ; p != str.end() && isspace(*p); ++p );
        // skip =
        for ( ; p != str.end() && *p == '='; ++p );
        // skip white spaces
        for ( ; p != str.end() && isspace(*p); ++p );
                
        // skip opening "
        for ( ; *p == '"' && p != str.end() ; ++p );
                
        // get param value
        insert_iterator<string> valueIt(paramValue, paramValue.begin());
        for ( ; p != str.end(); ++p)
        {
            if (*p == '\"')
                break;
            *valueIt = *p;
        }
                
    }
        
    // -----------------------------------------------------------------------
        
    void UnSerialization::parseParamOnly(const string& str, string& tagParam, string& paramValue)
    {
        // simple tag. Return just the string
        if ( str.find('=') == string::npos )
        {
            return;
        } 
                
        tagParam = "";
        paramValue = "";
                
        string::const_iterator p = str.begin();
                                
        // skip white spaces
        for ( ; isspace(*p) && p != str.end() ; ++p );
                
        // get param name
        insert_iterator<string> paramIt(tagParam, tagParam.begin());
        for ( ; p != str.end(); ++p)
        {
            if (*p == '=')
                break;
            *paramIt = *p;
        }
                
        // skip white spaces
        for ( ; p != str.end() && isspace(*p); ++p );
        // skip =
        for ( ; p != str.end() && *p == '='; ++p );
        // skip white spaces
        for ( ; p != str.end() && isspace(*p); ++p );
                
        // skip opening "
        for ( ; *p == '"' && p != str.end() ; ++p );
                
        // get param value
        insert_iterator<string> valueIt(paramValue, paramValue.begin());
        for ( ; p != str.end(); ++p)
        {
            if (*p == '\"')
                break;
            *valueIt = *p;
        }
                
    }
        
    // -----------------------------------------------------------------------

    bool UnSerialization::seekAndParseParamTag(nor_utils::StreamTokenizer& st, const string& tag, string& tagParam, string& paramValue)
    {
        bool tagFound = false;
        string rawTag;
        string foundTag;
        string::const_iterator p;
                
        tagParam = "";
        paramValue = "";
                
        do {
                        
            // the full tag. I.e. <tag param="val">
            rawTag = st.next_token();
            foundTag = "";
                        
            p = rawTag.begin();
                        
            // get tag name
            insert_iterator<string> tagIt(foundTag, foundTag.begin());
            for ( ; p != rawTag.end(); ++p)
            {
                if ( isspace(*p) )
                    break;
                *tagIt = *p;
            }
                        
            if ( nor_utils::cmp_nocase(tag, foundTag) )
            {
                tagFound = true;
                break;
            }
                        
        } while( st.has_token() );
                
        if ( !tagFound )
            return false;
                
        
        // skip white spaces
        for ( ; isspace(*p) && p != rawTag.end() ; ++p );
                
        // get param name
        insert_iterator<string> paramIt(tagParam, tagParam.begin());
        for ( ; p != rawTag.end(); ++p)
        {
            if (*p == '=')
                break;
            *paramIt = *p;
        }
                
        // skip white spaces
        for ( ; p != rawTag.end() && isspace(*p); ++p );
        // skip =
        for ( ; p != rawTag.end() && *p == '='; ++p );
        // skip white spaces
        for ( ; p != rawTag.end() && isspace(*p); ++p );
                
        // skip opening "
        for ( ; p != rawTag.end() && *p == '"'; ++p );
                
        // get param value
        insert_iterator<string> valueIt(paramValue, paramValue.begin());
        for ( ; p != rawTag.end(); ++p)
        {
            if (*p == '\"')
                break;
            *valueIt = *p;
        }
                
        return true;
    }
        
    // -----------------------------------------------------------------------

    bool UnSerialization::seekAndParseSpecificParamTag(nor_utils::StreamTokenizer& st, const string& tag, const string& targetTagParam, string& paramValue)
    {
        bool tagFound = false;
        string rawTag;
        string foundTag;
        string::const_iterator p;
                
        string tagParam;
        
        tagParam = "";
        paramValue = "";
                
        do {
                        
            // the full tag. I.e. <tag param="val">
            rawTag = st.next_token();
            foundTag = "";
                        
            p = rawTag.begin();
                        
            // get tag name
            insert_iterator<string> tagIt(foundTag, foundTag.begin());
            for ( ; p != rawTag.end(); ++p)
            {
                if ( isspace(*p) )
                    break;
                *tagIt = *p;
            }
                        
            if ( nor_utils::cmp_nocase(tag, foundTag) )
            {
                tagFound = true;
                break;
            }
                        
        } while( st.has_token() );
                
        if ( !tagFound )
            return false;
                
        while (targetTagParam.compare(tagParam) != 0 && p != rawTag.end()) {
            
            // skip white spaces
            for ( ; isspace(*p) && p != rawTag.end() ; ++p );
            
            // get param name
            insert_iterator<string> paramIt(tagParam, tagParam.begin());
            for ( ; p != rawTag.end(); ++p)
            {
                if (*p == '=')
                    break;
                *paramIt = *p;
            }
            
            // skip white spaces
            for ( ; p != rawTag.end() && isspace(*p); ++p );
            // skip =
            for ( ; p != rawTag.end() && *p == '='; ++p );
            // skip white spaces
            for ( ; p != rawTag.end() && isspace(*p); ++p );
            
            // skip opening "
            for ( ; p != rawTag.end() && *p == '"'; ++p );
            
            // get param value
            insert_iterator<string> valueIt(paramValue, paramValue.begin());
            for ( ; p != rawTag.end(); ++p)
            {
                if (*p == '\"')
                    break;
                *valueIt = *p;
            }
            
        }
                
        return true;
    }
        
    // -----------------------------------------------------------------------
    
    void UnSerialization::loadCascadeHypotheses(const string& shypFileName, 
                                                vector<vector<BaseLearner*> >& weakHypotheses, vector<AlphaReal>& thresholds,
                                                InputData* pTrainingData, int verbose)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }
                
        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
                
        // Move until it finds the multiboost tag
        if ( !seekSimpleTag(st, "cascade") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid Cascade Strong Hypothesis file!!" << endl;
            exit(1);
        }
                
        // Move until it finds the algo tag
        // Not used for the moment
        string algoName = seekAndParseEnclosedValue<string>(st, "algo");
                
        cout << endl;
                
        // Will be filled in by seekAndParseParamTag in each iteration
        for ( int stagei=0; ; ++stagei )
        {
            if ( !seekParamTag(st, "stage") )
                break;
                        
            vector<BaseLearner*> tmpVec(0);
            int weakHypNum = seekAndParseEnclosedValue<int>(st, "weakhypnum");
            double currentThreshold = seekAndParseEnclosedValue<double>(st, "threshold");
            thresholds.push_back(currentThreshold);
                        
            // move until the next weak hypothesis
            for(int i=0; i< weakHypNum; ++i)
            {
                if ( seekParamTag(st, "weakhyp") )
                    loadHypothesis(st, tmpVec, pTrainingData, verbose);
                else
                {
                    cout << "Warning!!! Weak learners are missing from " << stagei << ". stage!!" << endl;
                    break;
                }
            }
            cout << "Stage: " << stagei << " Weak hyp number: " << tmpVec.size() << endl;
            weakHypotheses.push_back( tmpVec );
        }                               
    }
        
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
        
        
} // end of namespace MultiBoost
