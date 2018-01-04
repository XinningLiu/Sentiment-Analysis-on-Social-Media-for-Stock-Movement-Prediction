#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
A multiprocessing version of running the sentiment analysis
'''



__author__ = 'Xinning Liu'
from multiprocessing import Pool

import os, time, random

import sys
if not("/Users/liuxinning/Documents/python" in sys.path):
    sys.path.append("/Users/liuxinning/Documents/python")

import sentimentAnalysis as SA

def LDAandJST(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    # train LDA
    if name == 0:        
        [ XOMphi_LDA,  XOMalpha_LDA, timeConsumption] = SA.trainLDA( XOMTrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("XOM LDA training spent",timeConsumption,"sec.")
        [ XOMtheta_LDA, timeConsumption] = SA.testLDA( XOMTestMessageList,  XOMphi_LDA,  XOMalpha_LDA, iter_Test, dictionary, index_dict)
        print("XOM LDA test spent",timeConsumption,"sec.")
        SA.writeTheta_LDA('XOM',XOMtheta_LDA)
    elif name == 1:
        [EBAYphi_LDA, EBAYalpha_LDA, timeConsumption] = SA.trainLDA(EBAYTrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("EBAY LDA training spent",timeConsumption,"sec.")
        [EBAYtheta_LDA, timeConsumption] = SA.testLDA(EBAYTestMessageList, EBAYphi_LDA, EBAYalpha_LDA, iter_Test, dictionary, index_dict)
        print("EBAY LDA test spent",timeConsumption,"sec.")
        SA.writeTheta_LDA('EBAY',EBAYtheta_LDA)
    elif name == 2:
        [   Tphi_LDA,    Talpha_LDA, timeConsumption] = SA.trainLDA(   TTrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("T LDA training spent",timeConsumption,"sec.")
        [   Ttheta_LDA, timeConsumption] = SA.testLDA(   TTestMessageList,    Tphi_LDA,    Talpha_LDA, iter_Test, dictionary, index_dict)
        print("T LDA test spent",timeConsumption,"sec.")
        SA.writeTheta_LDA('T',Ttheta_LDA)
    elif name == 3:
        [  BAphi_LDA,   BAalpha_LDA, timeConsumption] = SA.trainLDA(   BATrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("BA LDA training spent",timeConsumption,"sec.")
        [   BAtheta_LDA, timeConsumption] = SA.testLDA(   BATestMessageList,    BAphi_LDA,    BAalpha_LDA, iter_Test, dictionary, index_dict)
        print("BA LDA test spent",timeConsumption,"sec.")     
        SA.writeTheta_LDA('BA',BAtheta_LDA)
    elif name == 4:
        [AAPLphi_LDA, AAPLalpha_LDA, timeConsumption] = SA.trainLDA( AAPLTrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("AAPL LDA training spent",timeConsumption,"sec.")
        [ AAPLtheta_LDA, timeConsumption] = SA.testLDA( AAPLTestMessageList,  AAPLphi_LDA,  AAPLalpha_LDA, iter_Test, dictionary, index_dict)
        print("AAPL LDA test spent",timeConsumption,"sec.")
        SA.writeTheta_LDA('AAPL',AAPLtheta_LDA)
    elif name == 5:
        [NVDAphi_LDA, NVDAalpha_LDA, timeConsumption] = SA.trainLDA( NVDATrainMessageList,iter_Training,dictionary,index_dict,T_LDA)
        print("NVDA LDA training spent",timeConsumption,"sec.")
        [ NVDAtheta_LDA, timeConsumption] = SA.testLDA(NVDATestMessageList,  NVDAphi_LDA,  NVDAalpha_LDA, iter_Test, dictionary, index_dict)
        print("NVDA LDA test spent",timeConsumption,"sec.")
        SA.writeTheta_LDA('NVDA',NVDAtheta_LDA)
   # train JST
    elif name == 6:
        [ XOMphi_JST,  XOMalpha_JST,  XOMgamma, timeConsumption] = SA.trainJST( XOMTrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("XOM JST training spent",timeConsumption,"sec.")
        [ XOMtheta_JST, timeConsumption] = SA.testJST( XOMTestMessageList,  XOMphi_JST,  XOMalpha_JST,  XOMgamma, iter_Test, dictionary, index_dict)
        print("XOM JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('XOM',XOMtheta_JST)
    elif name == 7:
        [EBAYphi_JST, EBAYalpha_JST, EBAYgamma, timeConsumption] = SA.trainJST(EBAYTrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("EBAY JST training spent",timeConsumption,"sec.")
        [EBAYtheta_JST, timeConsumption] = SA.testJST(EBAYTestMessageList, EBAYphi_JST, EBAYalpha_JST, EBAYgamma, iter_Test, dictionary, index_dict)
        print("EBAY JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('EBAY',EBAYtheta_JST)
    elif name == 8:
        [   Tphi_JST,    Talpha_JST,    Tgamma, timeConsumption] = SA.trainJST(   TTrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("T JST training spent",timeConsumption,"sec.")
        [   Ttheta_JST, timeConsumption] = SA.testJST(   TTestMessageList,    Tphi_JST,    Talpha_JST,    Tgamma, iter_Test, dictionary, index_dict)
        print("T JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('T',Ttheta_JST)
    elif name == 9:
        [   BAphi_JST,    BAalpha_JST,    BAgamma, timeConsumption] = SA.trainJST(   BATrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("BA JST training spent",timeConsumption,"sec.")
        [   BAtheta_JST, timeConsumption] = SA.testJST(   BATestMessageList,    BAphi_JST,    BAalpha_JST,    BAgamma, iter_Test, dictionary, index_dict)
        print("BA JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('BA',BAtheta_JST)
    elif name == 10:
        [ AAPLphi_JST,  AAPLalpha_JST,  AAPLgamma, timeConsumption] = SA.trainJST( AAPLTrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("AAPL JST training spent",timeConsumption,"sec.")
        [ AAPLtheta_JST, timeConsumption] = SA.testJST( AAPLTestMessageList,  AAPLphi_JST,  AAPLalpha_JST,  AAPLgamma, iter_Test, dictionary, index_dict)
        print("AAPL JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('AAPL',AAPLtheta_JST)
    elif name == 11:
        [ NVDAphi_JST,  NVDAalpha_JST,  NVDAgamma, timeConsumption] = SA.trainJST( NVDATrainMessageList, iter_Training, dictionary, index_dict, S_JST, T_JST)
        print("NVDA JST training spent",timeConsumption,"sec.")
        [ NVDAtheta_JST, timeConsumption] = SA.testJST( NVDATestMessageList,  NVDAphi_JST,  NVDAalpha_JST,  NVDAgamma, iter_Test, dictionary, index_dict)
        print("NVDA JST test spent",timeConsumption,"sec.")
        SA.writeTheta_JST('NVDA',NVDAtheta_JST)


# main process


if __name__ == '__main__':
    startTime=time.time()

    print('Parent process %s.' % os.getpid())
    iter_Training = 1000

    T_LDA = 10

    S_JST = 3
    T_JST = 10

    iter_Test=1000


    # Read training set
    [ XOMTrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("XOM_20171122_training.txt.xml",     [2017,11,22],False)
    print("read XOM training set spent",timeConsumption,"sec.")
    print("XOM training set has",len(XOMTrainMessageList),"messages.")
    [EBAYTrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("EBAY_201711221430_training.txt.xml",[2017,11,22],False,dictionary)
    print("read EBAY training set spent",timeConsumption,"sec.")
    print("EBAY training set has",len(EBAYTrainMessageList),"messages.")
    [   TTrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("T_20171122_training.txt.xml",       [2017,11,22],False,dictionary)
    print("read T training set spent",timeConsumption,"sec.")
    print("T training set has",len(TTrainMessageList),"messages.")

    [  BATrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML(  "BA_201711231820.txt.xml",[2017,11,23],False,dictionary)
    print("read BA training set spent",timeConsumption,"sec.")
    print("BA training set has",len(BATrainMessageList),"messages.")
    [AAPLTrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("AAPL_201711222256.txt.xml",[2017,11,22],False)
    print("read AAPL training set spent",timeConsumption,"sec.")
    print("AAPL training set has",len(AAPLTrainMessageList),"messages.")
    [NVDATrainMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("NVDA_201711222148.txt.xml",[2017,11,22],False,dictionary)
    print("read NVDA training set spent",timeConsumption,"sec.")
    print("NVDA training set has",len(NVDATrainMessageList),"messages.")

    print()

    print("Dictionary has",len(dictionary),"words")


    print()
    # Read test set
    [ XOMTestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("XOM_20171122_test.txt.xml",     [2017,11,22],True,dictionary)
    print("read XOM test set spent",timeConsumption,"sec.")
    print("XOM test set has",len(XOMTestMessageList),"messages.")
    [EBAYTestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("EBAY_201711221430_test.txt.xml",[2017,11,22],True,dictionary)
    print("read EBAY test set spent",timeConsumption,"sec.")
    print("EBAY test set has",len(EBAYTestMessageList),"messages.")
    [   TTestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("T_20171122_test.txt.xml",       [2017,11,22],True,dictionary)
    print("read T test set spent",timeConsumption,"sec.")
    print("T test set has",len(TTestMessageList),"messages.")

    [   BATestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("BA_201711231820_test.txt.xml",       [2017,11,22],True,dictionary)
    print("read BA test set spent",timeConsumption,"sec.")
    print("BA test set has",len(BATestMessageList),"messages.")
    [ AAPLTestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("AAPL_201711222256_test.txt.xml",     [2017,11,22],True,dictionary)
    print("read AAPL test set spent",timeConsumption,"sec.")
    print("AAPL test set has",len(AAPLTestMessageList),"messages.")
    [ NVDATestMessageList, timeConsumption, dictionary, index_dict] = SA.readXML("NVDA_201711222148_test.txt.xml",[2017,11,22],True,dictionary)
    print("read NVDA test set spent",timeConsumption,"sec.")
    print("NVDA test set has",len(NVDATestMessageList),"messages.")


    print()


    # Train ABS
    print("Train ABS")

    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS( XOMTrainMessageList)
    print("XOM ABS training spent",timeConsumption,"sec.")
    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS(EBAYTrainMessageList, topic_dictionary)
    print("EBAY ABS training spent",timeConsumption,"sec.")
    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS(   TTrainMessageList, topic_dictionary)
    print("T ABS training spent",timeConsumption,"sec.")

    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS(   BATrainMessageList, topic_dictionary)
    print("BA ABS training spent",timeConsumption,"sec.")
    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS( AAPLTrainMessageList)
    print("AAPL ABS training spent",timeConsumption,"sec.")
    [topic_dictionary, index_topic_dict, timeConsumption] = SA.trainABS( NVDATrainMessageList, topic_dictionary)
    print("NVDA ABS training spent",timeConsumption,"sec.")

    print()
    print("The topic dictionary has",len(topic_dictionary),"topics.")
    print()
    # Test ABS
    print("Test ABS")
    [ XOMsenti_ABS,timeConsumption] = SA.testABS( XOMTestMessageList, topic_dictionary, index_topic_dict)
    print("XOM ABS test spent",timeConsumption,"sec.")
    [EBAYsenti_ABS,timeConsumption] = SA.testABS(EBAYTestMessageList, topic_dictionary, index_topic_dict)
    print("EBAY ABS test spent",timeConsumption,"sec.")
    [   Tsenti_ABS,timeConsumption] = SA.testABS(   TTestMessageList, topic_dictionary, index_topic_dict)
    print("T ABS test spent",timeConsumption,"sec.")

    [   BAsenti_ABS,timeConsumption] = SA.testABS(   BATestMessageList, topic_dictionary, index_topic_dict)
    print("BA ABS test spent",timeConsumption,"sec.")
    [ AAPLsenti_ABS,timeConsumption] = SA.testABS( AAPLTestMessageList, topic_dictionary, index_topic_dict)
    print("AAPL ABS test spent",timeConsumption,"sec.")
    [ NVDAsenti_ABS,timeConsumption] = SA.testABS(NVDATestMessageList, topic_dictionary, index_topic_dict)
    print("NVDA ABS test spent",timeConsumption,"sec.")
    print()


    # multiprocess
    # Train and test LDA and JST

    print("Train LDA and JST, iteration =", iter_Training)
    print("Train LDA with number of topics=",T_LDA)
    print("Train JST with number of sentencse =", S_JST, "number of topics=", T_JST)
    print("Test LDA and JST, iteration = ", iter_Test)

    print()





    pLDAJST = Pool(4)
    
    for i in range(12):
        pLDAJST.apply_async(LDAandJST, args=(i,))
    print('Waiting for all LDA and JST training and testing processes done...')
    pLDAJST.close()
    pLDAJST.join()
    print('All all LDA and JST training and testing done.')






    print()

    # Read theta matrix from file
    
    XOMtheta_LDA=SA.readTheta_LDA('XOM')
    XOMtheta_JST=SA.readTheta_JST('XOM')
    EBAYtheta_LDA=SA.readTheta_LDA('EBAY')
    EBAYtheta_JST=SA.readTheta_JST('EBAY')
    Ttheta_LDA=SA.readTheta_LDA('T')
    Ttheta_JST=SA.readTheta_JST('T')
    BAtheta_LDA=SA.readTheta_LDA('BA')
    BAtheta_JST=SA.readTheta_JST('BA')
    AAPLtheta_LDA=SA.readTheta_LDA('AAPL')
    AAPLtheta_JST=SA.readTheta_JST('AAPL')
    NVDAtheta_LDA=SA.readTheta_LDA('NVDA')
    NVDAtheta_JST=SA.readTheta_JST('NVDA')
    print()

    # Write sentiment analysis result

    [ XOMdailySentiment,timeConsumption] = SA.setDailySentiment( "XOM", XOMTestMessageList, XOMtheta_LDA, XOMtheta_JST, XOMsenti_ABS)

    [EBAYdailySentiment,timeConsumption] = SA.setDailySentiment("EBAY",EBAYTestMessageList,EBAYtheta_LDA,EBAYtheta_JST,EBAYsenti_ABS)

    [   TdailySentiment,timeConsumption] = SA.setDailySentiment(   "T",   TTestMessageList,   Ttheta_LDA,   Ttheta_JST,   Tsenti_ABS)

    [   BAdailySentiment,timeConsumption] = SA.setDailySentiment(   "BA",   BATestMessageList,   BAtheta_LDA,   BAtheta_JST,   BAsenti_ABS)

    [ AAPLdailySentiment,timeConsumption] = SA.setDailySentiment( "AAPL", AAPLTestMessageList, AAPLtheta_LDA, AAPLtheta_JST, AAPLsenti_ABS)

    [ NVDAdailySentiment,timeConsumption] = SA.setDailySentiment( "NVDA", NVDATestMessageList, NVDAtheta_LDA, NVDAtheta_JST, NVDAsenti_ABS)

    print("Use",time.time()-startTime,"sec to get all done")

