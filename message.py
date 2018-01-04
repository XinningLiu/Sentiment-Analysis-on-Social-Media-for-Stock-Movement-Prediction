Months=(0,31,28,31,30,31,30,31,31,30,31,30,31)
class Date(object):
    def __init__(self,year,month,day):
        self.year=year
        self.month=month
        self.day=day
        
    def __str__(self):
        return 'Date=%d.%d.%d' % (self.year,self.month,self.day)
        
    def daysago(self,days):
        if days<self.day:
            return Date(self.year,self.month,self.day-days)
        else:
            month=self.month
            day=self.day
            year=self.year
            while days>0:
                month=month-1
                if month==0:
                    month=12
                    year=year-1
                days=days-day
                day=Months[month]
                if month==2 and (year==2016 or year==2012):
                    if days<=day:
                        return Date(year,month,day-days+1)
                    else:
                        days=days-1
                if days<day:
                    return Date(year,month,day-days)
                

class Message(object):
    def __init__(self,idx):
        self.id=idx
#        self.date=Date(0,0,0)
#        self.NLPSentiment=0
#        self.LDASentiment=0
#        self.JSTSentiment=0
#        self.ASBSentiment=0
        self.words={}
        self.wordList=[]
        self.wordPOS=[]
    def setDaysAgo(self,daysago):
        self.daysAgo=daysago
        
    def setDate(self,date):
        self.date=date

    def setNLPSentiment(self,s):
        self.NLPSentiment=s

class ldaMessage(object):
    def __init__(self,n_topics):
        self.word=[]
        self.topic=[0 for x in range(n_topics)]
        self.n_topics=0

class myLDA(object):
    def __init__(self,n_topics,n_dict,n_messages):
        self.n_topics=n_topics
        self.n_dict=n_dict
        self.topic_word=[[0 for x in range(n_dict)] for y in range(n_topics)]
        self.topic_n_word=[0 for y in range(n_topics)]
        self.n_messages=n_messages
        self.messages=[ldaMessage(n_topics) for x in range(n_messages)]
        
class jstMessage(object):
    def __init__(self,n_sentiments,n_topics):
        self.word=[]
        self.n_sentiments=0
        self.sentiment=[0 for y in range(n_sentiments)]
        self.sentiment_topic=[[0 for x in range(n_topics)] for y in range(n_sentiments)]
        
class myJST(object):
    def __init__(self,n_sentiments,n_topics,n_dict,n_messages):
        self.n_topics=n_topics
        self.n_dict=n_dict
        self.n_sentiments=n_sentiments
        self.n_messages=n_messages
        self.message= \
            [jstMessage(n_sentiments,n_topics) for x in range(n_messages)]
        self.sentiment_topic_word= \
            [[[0 for x in range(n_dict)] for y in range(n_topics)] for z in range(n_sentiments)]
        self.sentiment_topic_n_word= \
            [[0 for y in range(n_topics)] for z in range(n_sentiments)]
        
class absMessage(object):
    def __init__(self,message):
        self.b_topic=[False for i in range(len(message.wordList))]
        self.topic=[]
        self.location=[]
        self.sentiment=[]
        self.n_topics=0

class myABS(object):
    def __init__(self, messagelist):
        self.message=[absMessage(message) for message in messagelist]
