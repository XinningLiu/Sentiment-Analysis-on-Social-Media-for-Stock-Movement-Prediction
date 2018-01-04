#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Read the NLP output xml file, check the sturcture. For test set, save it's NLP score
[messageList, timeConsumption, dictionary, index_dict] = readXML(s_file,d_currenttimestamp,b_test=T/F,dictionary={})

Train LDA, get the phi matrix
[phi_LDA, alpha, timeConsumption] = trainLDA(messagelist,iteration,dictionary,index_dict,T)
    T: number of Topics

Train JST, get the phi matrix
[phi_JST, alpha, gamma, timeConsumption] = trainJST(trainMessagelist,iteration,dictionary,index_dict,S,T)

Train ABS, get the topic dictionary
[topic_dictionary,index_topic_dict,timeConsumption] = trainABS(trainMessagelist,topic_dictionary={})


Find the topic distribution over the message in the test set
[theta_LDA,timeConsumption] = testLDA(testMessagelist, phi_LDA, alpha, iteration, dictionary, index_dict)

Find the sentiment-topic joint distribution over the message in the test set
[theta_JST,timeConsumption] = testJST(testMessagelist, phi_JST, alpha, gamma, iteration, dictionary, index_dict)

Find the ABS score of the message in the test set
[senti_ABS,timeConsumption] = testABS(testMessagelist, topic_dictionary, idx_dict):

Compute the daily NLP, LDA, JST and ABS score, also compute the daily importance score of topices in the topic dictionary
[dailySentiment ,timeConsumption] = setDailySentiment(s_name,testMessagelist,theta_LDA,theta_JST,senti_ABS)

Compute the word sentiment and ABS method
word_sentiment = score_WentiWorldNet(word)

For interprocesser communication, write theta matrix on dist, and read the matrix for computation
writeTheta_LDA(s_name,theta_LDA)
readTheta_LDA(s_name)
writeTheta_JST(s_name,theta_JST)
readTheta_LDA(s_name)
'''

__author__ = 'Xinning Liu'
import xml.etree.ElementTree as ET
from message import Date
from message import Message
from message import myLDA
from message import myJST
from message import myABS
from nltk.corpus import sentiwordnet as swn
import time
import random


posSet=set([  'CC', 'CD',  'DT',  'EX',  'FW',  'IN',  'JJ', 'JJR', 'JJS',
              'LS', 'MD',  'NN', 'NNP','NNPS', 'NNS', 'PDT', 'POS', 'PRP',
            'PRP$', 'RB', 'RBR', 'RBS',  'RP', 'SYM',  'TO',  'UH',  'VB',
             'VBD','VBG', 'VBN', 'VBP', 'VBZ', 'WDT',  'WP', 'WP$', 'WRB'])
posNounSet=set(['NN', 'NNP','NNPS', 'NNS'])


def readXML(s_file,d_currenttimestamp,b_test,dictionary={}):
    startTime=time.time()
    # initialization
#    timestamp=Date(d_currenttimestamp)
    s_path=r"/Users/liuxinning/Documents/python/data/"+s_file
    timestamp=Date(d_currenttimestamp[0],d_currenttimestamp[1],d_currenttimestamp[2])
    tree = ET.parse(s_path)
    root = tree.getroot()
    document=root.find("document")
    sentences=document.find("sentences")
    messagelength=0
    messagelist=[]
    messagesentiment=0
    sentenceslength=0
    
    for sentence in sentences:
        tokens=sentence.find("tokens")
        
        if (len(tokens)==2 and \
        (tokens[1].find('lemma').text=='!!' and   \
         tokens[0].find('lemma').text=='yesterday')) or \
         (len(tokens)==4 and \
          (tokens[3].find('lemma').text=='!!' and \
           tokens[2].find('lemma').text=='ago'))     or \
           (len(tokens)==3 and (tokens[2].find('lemma').text=='!!' and \
            (tokens[0].find('lemma').text=='last' and \
             (tokens[1].find('lemma').text=='year' or \
              tokens[1].find('lemma').text=='month')))):
        # Recognize a timestamp
            # set time stamp
            if len(tokens)==2 : 
                daysago=1
                date=timestamp.daysago(1)
            elif len(tokens)==3:
                if tokens[1].find('lemma').text=='year':
                    date=timestamp.daysago(366)
                    daysago=366
                else:
                    date=timestamp.daysago(31)
                    daysago=31
            else: #len(tokens)==4
                if tokens[1].find('lemma').text=='minute' or \
                   tokens[1].find('lemma').text=='hour':
                    date=timestamp
                    daysago=0
                else:
                    if tokens[1].find('lemma').text=='day':
                        scale=1
                    elif tokens[1].find('lemma').text=='month':
                        scale=31
                    else: #tokens[1].find('lemma').text=='year'
                        scale=366
                    daysago=int(tokens[0].find('lemma').text)*scale
                    date=timestamp.daysago(daysago)
                    
            # A message begins after a timestamp
            if messagelength==0:
                messagelength=1
                messagelist.append(Message(messagelength))
                messagelist[messagelength-1].setDate(date)
                messagelist[messagelength-1].setDaysAgo(daysago)
            elif sentenceslength>0:
                if b_test==True: # only test set has sentiment value
                    messagelist[messagelength-1].setNLPSentiment(messagesentiment/sentenceslength-2)
                    messagesentiment=0
                sentenceslength=0
                messagelength += 1
                messagelist.append(Message(messagelength))
                messagelist[messagelength-1].setDate(date)
                messagelist[messagelength-1].setDaysAgo(daysago)
            else:#last timesramp is empty
                messagelist[messagelength-1].setDate(date)
                messagelist[messagelength-1].setDaysAgo(daysago)
            continue # no need for message recognization

        # the sentence belong to a message
        b_nonEmpty=False
        for token in tokens:
            if token.find('POS').text in posSet:
                lemma=token.find('lemma').text.lower()
                if lemma in dictionary:
                    dictionary[lemma]+=1
                else:
                    if not b_test:
                        dictionary[lemma]=1
                    else:
                        continue

                b_nonEmpty=True                   

                if lemma in messagelist[messagelength-1].words:
                    messagelist[messagelength-1].words[lemma]+=1
                else:
                    messagelist[messagelength-1].words[lemma]=1
                messagelist[messagelength-1].wordList.append(lemma)
                
                # for Aspect-based sentiment only
                if token.find('POS').text in posNounSet:
                    pos=True
                else:
                    pos=False
                messagelist[messagelength-1].wordPOS.append(pos)
                
        if b_nonEmpty==True:
            sentenceslength += 1
            if b_test==True:
                sentimentValue=int(sentence.get('sentimentValue',-1))
                messagesentiment += sentimentValue
    # the last sentence in the set
    if sentenceslength>0:
        messagelist[messagelength-1].setNLPSentiment(messagesentiment/sentenceslength)
    else:
        messagelist.pop()

    i=0
    index_dict={}
    for word in sorted(dictionary):
        index_dict[word]=i
        i+=1

##    s_out=r"/Users/liuxinning/Documents/python/data/"+s_file+".output"
##    f_out=open(s_out,'w+',encoding='utf8')
##    for i in range(messagelength):
##        s=str(messagelist[i].daysAgo)+' days ago:\t'+' '.join(messagelist[i].wordList)+'.'+'\n'
##        t=f_out.write(s)
##    f_out.close()
    return [messagelist,time.time()-startTime,dictionary,index_dict]


def trainLDA(messagelist,iteration,dictionary,index_dict,T):
    startTime=time.time()
    W=len(dictionary)
    N=len(messagelist)
    alpha=50/T
    beta=200/W
    LDA=myLDA(T,W,N) 
    random.seed()

    for i in range(N):
        for word in messagelist[i].wordList:
            id_topic=random.randrange(T)
            LDA.messages[i].word.append(id_topic)
                # assign a topic to each word in message i
            LDA.messages[i].topic[id_topic]+=1
                # number of the id_topic appear in message[i] +1
            LDA.messages[i].n_topics+=1
                # number of topics in message i +1
            LDA.topic_word[id_topic][index_dict[word]]+=1
                # number of the word has id_topic +1
            LDA.topic_n_word[id_topic]+=1
                # number of words the id_topic appointed

    for j in range(iteration):
        for i in range(N):
            for k in range(len(messagelist[i].wordList)):
                word=messagelist[i].wordList[k]
                id_topic=LDA.messages[i].word[k]
                LDA.messages[i].topic[id_topic]-=1
                LDA.messages[i].n_topics-=1
                LDA.topic_word[id_topic][index_dict[word]]-=1
                LDA.topic_n_word[id_topic]-=1         

                prob_c=[(LDA.topic_word[x][index_dict[word]]+beta)*(LDA.messages[i].topic[x]+alpha)  \
                      /(LDA.topic_n_word[x]+beta*W) for x in range(T)]
                prob=[x/sum(prob_c) for x in prob_c]
                new_topic=random.choices(range(T),weights=prob)[0]
                LDA.messages[i].word[k]=new_topic
                LDA.messages[i].topic[new_topic]+=1
                LDA.messages[i].n_topics+=1
                LDA.topic_word[new_topic][index_dict[word]]+=1
                LDA.topic_n_word[new_topic]+=1
 #       print(j)

    phi_LDA=[[(LDA.topic_word[t][k]+beta)/(LDA.topic_n_word[t]+beta*W) for k in range(W)] for t in range(T) ]

    return [phi_LDA, alpha, time.time()-startTime]

def testLDA(messagelist, phi_LDA, alpha, iteration, dictionary, index_dict):
    startTime=time.time()
    T=len(phi_LDA)
    W=len(dictionary)
    N=len(messagelist)
    
    LDA=myLDA(T,W,N) 
    random.seed()
    for i in range(N):
        for word in messagelist[i].wordList:
            id_word=index_dict[word]
            prob_c=[phi_LDA[t][id_word] for t in range(T)]
            prob=[x/sum(prob_c) for x in prob_c]
            id_topic=random.choices(range(T),weights=prob)[0]
            LDA.messages[i].word.append(id_topic)
                # assign a topic to each word in message i
            LDA.messages[i].topic[id_topic]+=1
                # number of the id_topic appear in message[i] +1
            LDA.messages[i].n_topics+=1
                # number of topics in message i +1
            LDA.topic_word[id_topic][index_dict[word]]+=1
                # number of the word has id_topic +1
            LDA.topic_n_word[id_topic]+=1
                # number of words the id_topic appointed

    for j in range(iteration):
        for i in range(N):
            for k in range(len(messagelist[i].wordList)):
                word=messagelist[i].wordList[k]
                id_word=index_dict[word]

                id_topic=LDA.messages[i].word[k]
                LDA.messages[i].topic[id_topic]-=1
                LDA.messages[i].n_topics-=1
                LDA.topic_word[id_topic][index_dict[word]]-=1
                LDA.topic_n_word[id_topic]-=1         

                prob_c=[phi_LDA[t][id_word]*(LDA.messages[i].topic[t]+alpha) for t in range(T)]
                prob=[x/sum(prob_c) for x in prob_c]
                new_topic=random.choices(range(T),weights=prob)[0]
                LDA.messages[i].word[k]=new_topic
                LDA.messages[i].topic[new_topic]+=1
                LDA.messages[i].n_topics+=1
                LDA.topic_word[new_topic][index_dict[word]]+=1
                LDA.topic_n_word[new_topic]+=1
 #       print(j)

    theta_LDA=[[(LDA.messages[i].topic[t]+alpha)/(LDA.messages[i].n_topics+alpha*T) for t in range(T)] for i in range(N)]
    
    return [theta_LDA,time.time()-startTime]


def trainJST(messagelist,iteration,dictionary,index_dict,S,T):
    startTime=time.time()
    W=len(dictionary)
    N=len(messagelist)
    alpha=50/T
    beta=200/W
    gamma=1
    JST=myJST(S,T,W,N) 
    random.seed()

    for i in range(N):
        for word in messagelist[i].wordList:
            id_sentiment=random.randrange(S)
            id_topic=random.randrange(T)
        
            JST.message[i].word.append([id_sentiment,id_topic])
                # assign a topic and a sentiment to each word in message i
            JST.message[i].n_sentiments+=1
            JST.message[i].sentiment[id_sentiment]+=1
            JST.message[i].sentiment_topic[id_sentiment][id_topic]+=1

            JST.sentiment_topic_n_word[id_sentiment][id_topic]+=1
            JST.sentiment_topic_word[id_sentiment][id_topic][index_dict[word]]+=1

    for j in range(iteration):
        for i in range(N):
            for k in range(len(messagelist[i].wordList)):
                word=messagelist[i].wordList[k]
                id_sentiment=JST.message[i].word[k][0]
                id_topic=JST.message[i].word[k][1]

                JST.message[i].n_sentiments-=1
                JST.message[i].sentiment[id_sentiment]-=1
                JST.message[i].sentiment_topic[id_sentiment][id_topic]-=1

                JST.sentiment_topic_n_word[id_sentiment][id_topic]-=1
                JST.sentiment_topic_word[id_sentiment][id_topic][index_dict[word]]-=1

                prob_c=[[(JST.sentiment_topic_word[id_sentiment][id_topic][index_dict[word]]+beta)* \
                         (JST.message[i].sentiment_topic[id_sentiment][id_topic]+alpha)* \
                         (JST.message[i].sentiment[id_sentiment]+gamma)/ \
                         (JST.sentiment_topic_n_word[id_sentiment][id_topic]+beta*W)/ \
                         (JST.message[i].sentiment[id_sentiment]+T*alpha) \
                         for id_topic in range(T)] for id_sentiment in range(S)]
                scalar=sum([sum(x) for x in prob_c])
                prob=[[prob_c[id_sentiment][id_topic]/scalar for id_topic in range(T)] for id_sentiment in range(S)]

                new_sentiment=random.choices(range(S),weights=[sum(x) for x in prob])[0]
                new_topic = random.choices(range(T),weights=[x/sum(prob[new_sentiment]) for x in prob[new_sentiment]])[0]

                JST.message[i].word[k]=[new_sentiment,new_topic]
                JST.message[i].n_sentiments+=1
                JST.message[i].sentiment[new_sentiment]+=1
                JST.message[i].sentiment_topic[new_sentiment][new_topic]+=1

                JST.sentiment_topic_n_word[new_sentiment][new_topic]+=1
                JST.sentiment_topic_word[new_sentiment][new_topic][index_dict[word]]+=1
#        print(j)
    phi_JST=[[[(JST.sentiment_topic_word[id_sentiment][id_topic][id_word]+beta)/ \
               (JST.sentiment_topic_n_word[id_sentiment][id_topic]+beta*W) \
               for id_word in range(W)] for id_topic in range(T)] for id_sentiment in range(S)]          
    return [phi_JST, alpha, gamma, time.time()-startTime]

def testJST(messagelist, phi_JST, alpha, gamma, iteration, dictionary, index_dict):
    startTime=time.time()
    W=len(dictionary)
    N=len(messagelist)
    S=len(phi_JST)
    T=len(phi_JST[0])
    
    JST=myJST(S,T,W,N) 
    random.seed()

    for i in range(N):
        for word in messagelist[i].wordList:
            id_word= index_dict[word]

            prob_c=[sum([phi_JST[s][t][id_word] for t in range(T)]) for s in range(S)]
            prob=[x/sum(prob_c) for x in prob_c]
            
            id_sentiment=random.choices(range(S),weights=prob)[0]            

            prob_c=[sum([phi_JST[s][t][id_word] for s in range(S)]) for t in range(T) ]
            prob=[x/sum(prob_c) for x in prob_c]
            
            id_topic=random.choices(range(T),weights=prob)[0]
        
            JST.message[i].word.append([id_sentiment,id_topic])
                # assign a topic and a sentiment to each word in message i
            JST.message[i].n_sentiments+=1
            JST.message[i].sentiment[id_sentiment]+=1
            JST.message[i].sentiment_topic[id_sentiment][id_topic]+=1

            JST.sentiment_topic_n_word[id_sentiment][id_topic]+=1
            JST.sentiment_topic_word[id_sentiment][id_topic][index_dict[word]]+=1

    for j in range(iteration):
        for i in range(N):
            for k in range(len(messagelist[i].wordList)):
                word=messagelist[i].wordList[k]
                id_word= index_dict[word]
                id_sentiment=JST.message[i].word[k][0]
                id_topic=JST.message[i].word[k][1]

                JST.message[i].n_sentiments-=1
                JST.message[i].sentiment[id_sentiment]-=1
                JST.message[i].sentiment_topic[id_sentiment][id_topic]-=1

                JST.sentiment_topic_n_word[id_sentiment][id_topic]-=1
                JST.sentiment_topic_word[id_sentiment][id_topic][index_dict[word]]-=1

                prob_c=[[phi_JST[id_sentiment][id_topic][id_word] * \
                         (JST.message[i].sentiment_topic[id_sentiment][id_topic]+alpha)* \
                         (JST.message[i].sentiment[id_sentiment]+gamma)/ \
                         (JST.message[i].sentiment[id_sentiment]+T*alpha) \
                         for id_topic in range(T)] for id_sentiment in range(S)]
                scalar=sum([sum(x) for x in prob_c])
                prob=[[prob_c[id_sentiment][id_topic]/scalar for id_topic in range(T)] for id_sentiment in range(S)]

                new_sentiment=random.choices(range(S),weights=[sum(x) for x in prob])[0]
                new_topic = random.choices(range(T),weights=[x/sum(prob[new_sentiment]) for x in prob[new_sentiment]])[0]

                JST.message[i].word[k]=[new_sentiment,new_topic]
                JST.message[i].n_sentiments+=1
                JST.message[i].sentiment[new_sentiment]+=1
                JST.message[i].sentiment_topic[new_sentiment][new_topic]+=1

                JST.sentiment_topic_n_word[new_sentiment][new_topic]+=1
                JST.sentiment_topic_word[new_sentiment][new_topic][index_dict[word]]+=1
#        print(j)
    theta_JST=[[[ (JST.message[d].sentiment_topic[k][j]+alpha)/(JST.message[d].sentiment[k]+alpha*T) for j in range(T)] for k in range(S)] for d in range(N)]
    pi_JST=[[(JST.message[d].sentiment[k]+gamma)/(JST.message[d].n_sentiments+gamma*S) for k in range(S)]for d in range(N)]

    rtheta=[[[theta_JST[d][k][j]*pi_JST[d][k] for j in range(T)] for k in range(S)] for d in range(N)]


#    return [theta_JST,time.time()-startTime]
    return [ rtheta, time.time()-startTime]

def trainABS(messagelist,topic_dictionary={}):
    startTime=time.time()
    for message in messagelist:
        i=0
#        l=0
#        print(len(message.wordList))
        while i<len(message.wordList):
#            print(i,l)
            if message.wordPOS[i]==True:
                l=1
                while i+l< len(message.wordPOS):
                    if message.wordPOS[i+l]==True:
                        l+=1
                    else:
                        break
                s=' '.join(message.wordList[i:i+l])
                if s in topic_dictionary:
                    topic_dictionary[s]+=1
                else:
                    topic_dictionary[s]=1
                i+=l+1
            else:
                i+=1
#        print(message.id)

    topic_dictionary = {k:v for k,v in topic_dictionary.items() if v>=10}

    i=0
    index_dict={}
    for topic in sorted(topic_dictionary):
        index_dict[topic]=i
        i+=1

    return [topic_dictionary,index_dict,time.time()-startTime]


def score_SentiWorldNet(word):
    sentiList=list(swn.senti_synsets(word))
    if sentiList==[]:
        return 0
    else:
        try:
            score=(sentiList[0].pos_score()-sentiList[0].neg_score())/(sentiList[0].pos_score()+sentiList[0].neg_score())
            return (sentiList[0].pos_score()-sentiList[0].neg_score())/(sentiList[0].pos_score()+sentiList[0].neg_score())
        except ZeroDivisionError:
            return 0

def testABS(messagelist, topic_dictionary, idx_dict):
    startTime=time.time()
    ABS=myABS(messagelist)
    for j in range(len(messagelist)):
        message = messagelist[j]
        i=0
        while i<len(message.wordList):
            if message.wordPOS[i]==True:
                ABS.message[j].b_topic[i]=True
                l=1
                while i+l<len(message.wordList):
                    if message.wordPOS[i+l]==True:
                        ABS.message[j].b_topic[i+l]=True
                        l+=1
                    else:
                        break

                s=' '.join(message.wordList[i:i+l])
                if s in topic_dictionary:
                    ABS.message[j].topic.append(idx_dict[s])
                    ABS.message[j].location.append([i,i+l-1])
                
                i=i+l+1
            else:
                i+=1
                
        ABS.message[j].n_topics=len(ABS.message[j].topic)
        if ABS.message[j].n_topics==0:
            continue
        
        ABS.message[j].sentiment=[0 for k in range(ABS.message[j].n_topics)]

        for i in range(len(message.wordList)):
            if not ABS.message[j].b_topic[i]:
                score=score_SentiWorldNet(message.wordList[i])
                if score==0:
                    continue
                for k in range(ABS.message[j].n_topics):
                    if i< ABS.message[j].location[k][0]:
                        ABS.message[j].sentiment[k]+=score/(ABS.message[j].location[k][0]-i)
                    elif i> ABS.message[j].location[k][1]:
                        ABS.message[j].sentiment[k]+=score/(i-ABS.message[j].location[k][1])
#        print(j)

    senti_ABS=[[0 for l in range(len(topic_dictionary))] for j in range(len(messagelist))]

#    print(len(topic_dictionary))
#    for k,v in idx_dict.items():
#        print(k,':',v)
    for j in range(len(messagelist)):
        for k in range(ABS.message[j].n_topics):
#            print(ABS.message[j].topic[k])
            senti_ABS[j][ABS.message[j].topic[k]]+=ABS.message[j].sentiment[k]
    
    return [senti_ABS,time.time()-startTime]

def writeTheta_LDA(s_name,theta_LDA):

    s_out=r"/Users/liuxinning/Documents/python/data/"+s_name+"_theta_LDA.output"
    f=open(s_out,'w+')
    l=len(theta_LDA)
    m=len(theta_LDA[0])
    s=str(l)+'\n'
    f.write(s)
    s=str(m)+'\n'
    f.write(s)
    
    for i in range(l):
        s=''
        for j in range(m):
            s+=('\t'+str(theta_LDA[i][j]))
        s+='\n'
        t=f.write(s)
    f.close()

def readTheta_LDA(s_name):
    s_in=r"/Users/liuxinning/Documents/python/data/"+s_name+"_theta_LDA.output"
    f=open(s_in,'r')
    m = int(f.readline())
    n = int(f.readline())
    matrix=[[0 for i in range(n)] for j in range(m)]
    for j in range(m):
        s=f.readline()
        words=s.split()
        matrix[j] = [float(word) for word in words]
    f.close()
    return matrix

def writeTheta_JST(s_name,theta_JST):
    s_out=r"/Users/liuxinning/Documents/python/data/"+s_name+"_theta_JST.output"
    f=open(s_out,'w+')
    l=len(theta_JST)
    m=len(theta_JST[0])
    n=len(theta_JST[0][0])
    f=open(s_out,'w+')
    s=str(l)+'\n'
    f.write(s)
    s=str(m)+'\n'
    f.write(s)
    s=str(n)+'\n'
    f.write(s)
    for i in range(l):
        for j in range(m):
            s=''
            for k in range(n):
                s+=('\t'+str(theta_JST[i][j][k]))
            s+='\n'
            t=f.write(s)
    f.close()
def readTheta_JST(s_name):
    s_in=r"/Users/liuxinning/Documents/python/data/"+s_name+"_theta_JST.output"
    f=open(s_in,'r')
    l = int(f.readline())
    m = int(f.readline())
    n = int(f.readline())
    matrix=[[[0 for i in range(n)] for j in range(m)] for k in range(l)]
    for k in range(l):
        for j in range(m):
            s=f.readline()
            words=s.split()
            matrix[k][j] = [float(word) for word in words]
    f.close()
    return matrix

def setDailySentiment(s_name, messagelist,theta_LDA,theta_JST,senti_ABS):
    startTime=time.time()

    T_LDA=len(theta_LDA[0])

    S_JST=len(theta_JST[0])
    T_JST=len(theta_JST[0][0])

    T_ABS=len(senti_ABS[0])
    
    dailySentiment=[]
    il=0
    ir=0
    daysAgo=messagelist[ir].daysAgo
    dateList=[messagelist[ir].date]

    while ir<len(messagelist):
        if messagelist[ir].daysAgo==daysAgo:
            ir+=1
            continue
        
        n_messages=ir-il
        scoreNLP=sum([messagelist[i].NLPSentiment for i in range(il,ir)])/n_messages
        
        scoreLDA=[sum([theta_LDA[i][t] for i in range(il,ir)])/n_messages for t in range(T_LDA)]
        
        scoreJST=[[sum([theta_JST[i][s][t] for i in range(il,ir)])/n_messages for t in range(T_JST)] for s in range(S_JST)]

        scoreABS=[sum([senti_ABS[i][t] for i in range(il,ir)])/n_messages for t in range(T_ABS)]
        iscoreABS=[sum([senti_ABS[i][t]>0 for i in range(il,ir)])/n_messages for t in range(T_ABS)]

        dailySentiment.append([scoreNLP,scoreLDA,scoreJST,[scoreABS,iscoreABS]])
        daysAgo=messagelist[ir].daysAgo
        dateList.append(messagelist[ir].date)
        il=ir

    s_out=r"/Users/liuxinning/Documents/python/data/"+s_name+"_sentiment.output"
    f_out=open(s_out,'w+',encoding='utf8')
    t=f_out.write("NLP score is a scalor\n")
    t=f_out.write("LDA score is a vector with the ith entry is the score of topic i\n")
    t=f_out.write("JST score is a matrix with the [i,j]th entry (the ith row and j th colunm) is the score of sentiment i topic j\n")
    t=f_out.write("ABS score has two vectors with the ith entry of the first row is the score of topic i,\n the ith entry of the second row is the importance score of topic i\n")

    
    for i in range(len(dailySentiment)):
        s="Date\n\t%04d.%02d.%02d\n" % (dateList[i].year,dateList[i].month,dateList[i].day)

        t=f_out.write(s)

        t=f_out.write("NLP score\n")
        s='\t'+str(dailySentiment[i][0])+'\n'
        t=f_out.write(s)

        t=f_out.write("LDA score\n")
        s=''
        for t in range(len(dailySentiment[i][1])):
            s+='\t'+str(dailySentiment[i][1][t])
        s+='\n'
        t=f_out.write(s)

        t=f_out.write("JST score\n")
        for k in range(len(dailySentiment[i][2])):
            s=""
            for t in range(len(dailySentiment[i][2][k])):
                s+='\t'+str(dailySentiment[i][2][k][t])
            s+='\n'
            t=f_out.write(s)

        t=f_out.write("ABS score\n")
        s=""
        for t in range(len(dailySentiment[i][3][0])):
            s+='\t'+str(dailySentiment[i][3][0][t])
        s+="\n"

        for t in range(len(dailySentiment[i][3][1])):
            s+='\t'+str(dailySentiment[i][3][1][t])
        s+="\n"    
        t=f_out.write(s)
    f_out.close()

    
    return [dailySentiment ,time.time()-startTime] 
