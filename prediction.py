from flask import Flask, jsonify, render_template,request 
from sklearn import *
import textblob
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import punkt
import pandas as pd
import numpy as np
from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import recall_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import warnings  
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from scipy.stats import randint 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

"""
########################################
function to remove punctuation from text
########################################
""" 
def remove_punctuation(df, col):
    
    df[col] = df[col].str.replace('[^\w\s]','')
    return df
"""
############################################
function to convert everything to lower case
############################################
""" 
def convert_to_lowercase(df, col):
    df[col] = df[col].apply(lambda x: x.lower())
    return df

    """
############################
function to remove stopwords
############################
""" 
def remove_stopwords(df, col):
    ## dictionary_of_df_words ##  
    #NUMBER OF STOP WORDS#
    """
    stopWords = set(stopwords.words('english'))
    
    wordsFiltered = []
    for w in dictionary_of_df_words :
        if w not in stopWords:
            wordsFiltered.append(w)
    print(wordsFiltered)
    """
    """
    #checking if token words of the sentence
        #are in stopWOrds
        #if they are, then remove the words
        #form the string
    """
    i=0
    stopWords = set(stopwords.words('english'))
    l=[]
    for x in df[col]:
        l_of_words=word_tokenize(x)#.decode('utf-8'))
        wordsFiltered = []
        for w in l_of_words:
            if w not in stopWords:
                wordsFiltered.append(w)
        
        x=" ".join(wordsFiltered)
        #print "x is",x
        l.append(x)
        i+=1;
    
    #print "length of l is", len(l[0])
    #print "length of df is", df.shape
    df[col]=l
    return df
"""
##################################
function to perform lemmatization
##################################
""" 
def lemmatize(df, col):
    df['lem_comments'] = df[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df

"""
##################################
function to perform class imbalance
##################################
""" 
def solve_class_imbalance(b, notb):
    #get number of comments of bully dataset
    num_of_bully_com=list(b.shape)[0]
    
    #get 1.5 times non bully comments
    num_of_nonbully_com=int(num_of_bully_com*1.5)
    
    #now we sample num_of_nonbully_com from notb dataset 
    notb=notb.sample(n=num_of_nonbully_com)
    
    #concatenate b and notb
    concatb=pd.concat([b, notb], ignore_index=True)
    
    return concatb

"""
################################################
##function to get the list of words in the comment
################################################
""" 

def make_my_words_dictionary(df, col):
    t=list(df[col])
    dictionary_of_words=[]
    for x in t:
        words = word_tokenize(x.decode('utf-8'))
        #words=x.split()
        dictionary_of_words.append(words)
    
    dictionary_of_words=reduce(lambda x,y: x+y,dictionary_of_words)
    #print dictionary_of_words
    return dictionary_of_words

"""
##################################
function to perform tokenization
##################################
""" 
def tokenize(df, col):
    #tokenize
    dictionary_of_words=make_my_words_dictionary(df,"comment_text")
    
    return  dictionary_of_words

"""
##################################
function to remove special characters
##################################
""" 
def remove_special_char(df, col):
    #tokenize
    df[col] = df[col].str.replace('\W', ' ')
    #df[col]=df[col].apply(lambda x: " ".join(word) for word in x)
    #print(df[col])
    return df

"""
############################################
#function to perform pre-processing of data
############################################
"""
def perform_pre_processing(df, type_of_df, colu):
    # print ("\n pre processing",type_of_df,"dataset")
    """
    TRYING:
    
    removing punctuation from comment text
    """
    # print ("removing punctuation")
    # df= remove_punctuation(df,colu)
    
    """
    
    removing special char
    """
    # print("removing special char")
    #df= remove_special_char(df, colu)
    
    """
    
    TRYING:
    
    converting everything to lower case
    """
    # print ("converting to lower case")
    df= convert_to_lowercase(df,colu)
    
    """
    
    TRYING:
        
        removing stop words
        
    """
    # print ("removing stop words")
    df=remove_stopwords(df, colu)
        
    """
    TRYING:
    (preferred over stemming)
    Lemmatization: onverts the word into its root word
    """
    # print ("lemmatizing")
    df= lemmatize(df, colu)

    

    
    #return df, tf1,dictionary_of_df_words
    return df

"""
############################################
#function to bag of words of data
############################################
"""
def make_bag_of_words(df, col, df_test):#), tf1):
    # print("\n making bow")
    """
    TRYING:
    
    Bag of Words:representation of text which describes the presence of words within the text data.
    
    two similar text fields will contain similar kind of words, and will therefore have a similar bag of words. Further, 
    that from the text alone we can learn something about the meaning of the document.
        
    """
    #bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

    """
    TFIDF 
    
    """
    bow = TfidfVectorizer()
    
    #["ques","ans"]
    #train_bow = bow.fit_transform(df[col])
    
    train_bow = bow.fit_transform(df[col])
    
    #test_bow=bow.transform(df_test[col])
    
    test_bow=bow.transform(df_test[col])
    
    #print(train_bow.toarray())
    
    #print(len(train_bow.toarray()))
    #gives names of features
    #print "The feature names are"
    #print(bow.get_feature_names())
    
    X_train=train_bow.toarray()
    X_test=test_bow.toarray()
    
    # get all unique words in the corpus
    vocab = bow.get_feature_names_out()
    # show document feature vectors
    BOW=pd.DataFrame(X_train)#, columns=vocab)
    BOW.columns=vocab
    
    #Feature Engineering both training and test
    #X_train=perform_feature_engineering(X_train)
    #X_test=perform_feature_engineering(X_test)
    
    """
    TFIDVectorizer
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df['lem_comments'])
    print(vectorizer.get_feature_names())
    
    test_bow=vectorizer.fit_transform(df_test['comment_text'])
    X_test=test_bow.toarray()
    """
    
    ##################
    return X_train, X_test, BOW, bow

"""
############################### 
function to Fit the Model and find accuracy 
############################### 
"""
def make_diff_models(type_of_model, random='No', grid='No'):


    if type_of_model=='logistic':
        """# Logistic Regression #"""
        if random=='No' and grid=='No':
            #clf = LogisticRegression(random_state=0).fit(X_train, y_train)
            clf = LogisticRegression().fit(X_train, y_train)
        
        elif random=='Yes':
            #### Random Search CV ###
            parameters={'penalty':['l1','l2'],'solver':['liblinear'],'C':np.logspace(-4, 4, 20)}
            model=LogisticRegression()
            clf = RandomizedSearchCV(model, parameters).fit(X_train, y_train)
    
        else:
            ####Grid Search CV ###
            parameters={'penalty':['l1','l2'],'solver':['liblinear'],'C':np.logspace(-4, 4, 20)}
            model=LogisticRegression()
            clf = GridSearchCV(model, parameters).fit(X_train, y_train)
        
        
    
    
    if type_of_model=='randomforest':
    
        if random=='No' and grid=='No':
            #clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
            clf = RandomForestClassifier().fit(X_train, y_train)
        
        elif random=='Yes':
            #### Random Search CV ###
            parameters={'n_estimators':list(range(10,101,10)),'max_features': list(range(6,32,5))}
            model=RandomForestClassifier()
            clf = RandomizedSearchCV(model, parameters).fit(X_train, y_train)
        
        else: 
            ###Grid Search CV ###
            parameters={'n_estimators':list(range(10,101,10)),'max_features': list(range(6,32,5))}
            model=RandomForestClassifier()
            clf = GridSearchCV(model, parameters).fit(X_train, y_train)
            
    
    if type_of_model=='svm':
        """# SVM #"""
        if random=='No' and grid=='No':
            #clf = SVC(gamma='auto').fit(X_train, y_train)
            clf = SVC().fit(X_train, y_train)
    
        elif random=='Yes':
            ###Random Search CV with SVM##
            parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            svc = SVC()
            clf = RandomizedSearchCV(svc, parameters).fit(X_train, y_train)
        
        else:
            ###Grid Search CV with SVM###
            parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            svc = SVC()
            clf = GridSearchCV(svc, parameters).fit(X_train, y_train)
    
    if type_of_model=='decisiontrees':
        """# Decision Tree #"""
        # Creating the hyperparameter grid  
        param_dist = {"max_depth": [3, None], 
                "max_features": randint(1, 9), 
                "min_samples_leaf": randint(1, 9), 
                "criterion": ["gini", "entropy"]} 

        # Instantiating Decision Tree classifier 
        tree = DecisionTreeClassifier() 
        
        # Instantiating RandomizedSearchCV object 
        clf = RandomizedSearchCV(tree, param_dist, cv = 5).fit(X_train, y_train)
        
    
    if type_of_model=='neuralnetwork':
        """# Neural Network (MLP)#"""
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)

    return clf

df=pd.read_csv("formspring_data.csv", delimiter="\t",encoding="utf-8")#delimiter="<br>")

#final columns for the new dataframe
col = list(df.columns)
col = col[0].split("\t")

l = []

for i in range(len(df)):
    x = list(df.iloc[i])
    
    # Check if the data is not None before splitting
    if x[0] is not None:
        x = x[0].split("\t")
        
        # Check if the number of columns in x matches the number of columns in col
        if len(x) == len(col):
            l.append(x)
        else:
            print(f"Ignoring row {i} due to column mismatch: {x}")
            
# Create a new DataFrame with the columns
df1 = pd.DataFrame(l, columns=col)


#remove digits and special char from post
t=[]
for y in df['ques']:
    
    if isinstance(y, str):
        z=y.split(" ")
        #print("z is",z)
        ap=[]
        for x in z:
            alphanumeric = [character for character in x if character.isalnum()]
            alphanumeric = [character for character in x if not character.isdigit()]
            alphanumeric = "".join(alphanumeric)
            #print("alph is:", alphanumeric)
            ap.append(alphanumeric)
        ap=" ".join(ap)
        t.append(ap)
    else:
        t.append(x)
    
df['ques']=t

# Putting back apostrophe's may help
# df1['ques'] = df1['ques'].apply(lambda x: str(x) if isinstance(x, (str, int, float,)) else "")

# Now you can use the .str.replace method
df['ques'] = df['ques'].str.replace("&#039;", "'")

#remove others
df.ques = df.ques.str.replace("<br>", "") 
df.ques = df.ques.str.replace("&quot;", "") 
df.ques = df.ques.str.replace("<3", "") 

"""
3 occurrences of LABELDATA:
    ANSWER - YES or NO as to whether the post contains cyberbullying
    CYBERBULLYINGWORK - word(s) or phrase(s) identified by the mechanical turk worker as the reason it was tagged as cyberbullying (n/a or blank if no cyberbullying detected)
    SEVERITY - cyberbullying severity from 0 (no bullying) to 10 
    OTHER - other comments from the mechanical turk worker
    WORKTIME - time needed to label the post (in seconds)
    WORKER - mechanical turk worker id
    
"""

#ans1: yes==> 1038, no==> 11693
#ans2: yes==> 1005, no==> 11696

#get all the data where answer1 is bullying in nature
bully = df[df.ans1 == "Yes"].reset_index(drop=True)
not_bully=df[df.ans1 == "No"].reset_index(drop=True)

#df1[df1.ans1 == "Yes"].head()

"""
###########
MAIN
###########
"""    
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout


""" Solve class imbalance """
# Assign the result of solve_class_imbalance to concatb
concatb = solve_class_imbalance(bully, not_bully)

# Save the DataFrame with the updated decision column to a CSV file
concatb['decision'] = concatb['ans']
concatb.to_csv("updated_formspring_data.csv", index=True)

# Select specific columns from the concatenated DataFrame
selected_cols = ["ques", "ans", "ans1"]
concatb = concatb[selected_cols]

    
X = concatb[["ques","ans"]]#.values
y = concatb["ans1"]#.values

""" Divide into train and test dataset"""
#y=concatb['ans1']
#del concatb['ans1']
#X=concatb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  

"""###Pre processing training data###"""
#X_train=perform_pre_processing(X_train,'train','ans')
X_train=perform_pre_processing(X_train,'train','ques')


"""###Pre processing test data###"""
#X_test=perform_pre_processing(X_test,'test','ans')
X_test=perform_pre_processing(X_test,'test','ques')
"""###Making Bag of words###"""
#X_train, X_test, BOW=make_bag_of_words(X_train,'ans', X_test)#), tf1):
X_train, X_test, BOW, vectorizer=make_bag_of_words(X_train,'ques', X_test)#), tf1):

# X_train_text = X_train['ques'].values
# X_test_text = X_test['ques'].values

# # Preprocessing, tokenization, and padding for LSTM
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train_text)
# X_train_seq = tokenizer.texts_to_sequences(X_train_text)
# X_test_seq = tokenizer.texts_to_sequences(X_test_text)
# X_train = pad_sequences(X_train_seq, padding='post', maxlen=100)
# X_test = pad_sequences(X_test_seq, padding='post', maxlen=100)

# y_train = y_train.apply(lambda x: 1 if x == 'Yes' else 0)
# y_test = y_test.apply(lambda x: 1 if x == 'Yes' else 0)
# # Define and Compile the LSTM Model
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100))
# model.add(LSTM(64, return_sequences=True))
# # model.add(Dropout(0.2))   #add on
# model.add(LSTM(64))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) #add on
# model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True,save_weights_only=True, monitor='val_loss')



# # Train the LSTM Model
# model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

# # Save the LSTM Model
# model.save('cyberbullying_lstm_model.h5')

# # Make Predictions and Evaluate
# p = model.predict(X_test, verbose=1)
# predicted = [int(round(x[0])) for x in p]
# actual = y_test

# acc = accuracy_score(actual, predicted)
# prec = precision_score(actual, predicted, average='macro')
# rec = recall_score(actual, predicted, average='macro')
# f1 = f1_score(actual, predicted, average='weighted')

# print("\n The accuracy is", acc)
# print("\n The precision is", prec)
# print("\n The recall is", rec)
# print("\n The f1 score is", f1)

# target_names = ['Yes', 'No']
# print("\n Classification Report:")
# print(classification_report(actual, predicted, target_names=target_names))

# Save the Tokenizer for future use
# import joblib
# filename = 'tokenizer_cyber_bullying.pkl'
# with open(filename, 'wb') as handle:
#     joblib.dump(tokenizer, handle)



""" ### Make model and fit data ### """
type_of_model='logistic'
type_of_model='randomforest'
type_of_model='decisiontrees'
type_of_model='svm'
# type_of_model='neuralnetwork'

clf=make_diff_models(type_of_model, random='No', grid='No')

#Saving model
import joblib
filename = 'model_cyber_bullying.sav'
joblib.dump(clf, filename)
# pickle.dump(clf, open(filename, 'wb'))

"""# Evaluate #"""

#get test predictions
y_pred=clf.predict(X_test)

#find accuracy
acc=accuracy_score(y_test, y_pred)

print("\n The accuracy is", acc)
# print("\n")

#find precision
prec=precision_score(y_test, y_pred, average='macro')

print("\n The precision is", prec)
# print("\n")

# #find recall
rec=recall_score(y_test, y_pred, average='macro')

print("\n The recall is", rec)
# print("\n")

# #find f1 score
f1=f1_score(y_test, y_pred, average='weighted')

print("\n The f1 score is", f1)
print("\n")

# #get classification score
target_names = ['Yes','No']
# print("\n Classification Report:")
# print(classification_report(y_test, y_pred, target_names=target_names))
# print("\n")
# vectorizer = joblib.load('vectorizer_cyber_bullying.sav')

#print("best score:", clf.best_score_)
# print("\n Testing the phrase: ", arg0)
# v1=vectorizer.transform([arg0])
# text_to_predict = np.array(arg0)

# Preprocess the text using the vectorizer
# text_to_predict = vectorizer.transform([text_to_predict])

# Predict the class label
# ans = list(clf.predict(v1))[0]



# print("\n Testing the phrase: ", 'i am a normal girl')
v1=vectorizer.transform(['i am a normal girl'])
# v1=vectorizer.transform(['stupid bitch!, fuck off!'])
# v1=vectorizer.transform(['Fuck yes I am a good friend'])

print("Is it considered bullying?",list(clf.predict(v1))[0])

filename = 'vectorizer_cyber_bullying.sav'
joblib.dump(vectorizer, filename)
# arg2 = int(output["arg2"])
# ans = arg0
# anss = str(ans)


