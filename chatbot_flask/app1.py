intents = {"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
         "responses": ["Hello, thanks for visiting", "Hello,Good to see you again", "Hi there, how can I help?"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye","cu","bye"],
         "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {"tag": "hours",
         "patterns": ["What hours are you open?", "What are your hours?", "When are you open?", "Store opening time?" ],
         "responses": ["We're open every day 9am-9pm", "Our hours are 9am-9pm every day"]
        },
        {"tag": "mopeds",
         "patterns": ["Which mopeds do you have?", "What kinds of mopeds are there?", "What do you rent?" ],
         "responses": ["We rent Yamaha, Piaggio and Vespa mopeds", "We have Piaggio, Vespa and Yamaha mopeds"]
        },
        {"tag": "payments",
         "patterns": ["Do you take credit cards?", "Do you accept Mastercard?", "Are you cash only?" ],
         "responses": ["We accept VISA, Mastercard and AMEX", "We accept most major credit cards"]
        },
        {"tag": "opentoday",
         "patterns": ["Are you open today?", "When do you open today?", "What are your hours today?"],
         "responses": ["We're open every day from 9am-9pm", "Our hours are 9am-9pm every day"]
        },
        {"tag": "rental",
         "patterns": ["Can we rent a moped?", "I'd like to rent a moped", "How does this work?" ],
         "responses": ["Are you looking to rent today or later this week?"],
         "context_set": "rentalday"
        },
        {"tag": "today",
         "patterns": ["today"],
         "responses": ["For rentals today please call 1-800-MYMOPED", "Same-day rentals please call 1-800-MYMOPED"],
         "context_filter": "rentalday"
        },
        {"tag": "help",
         "patterns": ["hi,I need help"],
         "responses": ["Hi, How can I help you?"],
         "context_filter": ""
        },
        {"tag": "rate",
         "patterns": ["what is the exchange rate for USD today ?","exchange rates?", "present rates", "1 BHD how much USD?", "Todays Rate","What is the rate in USD?"],
         "responses": ["Buying rate is 0.375BHD and selling rate is 0.378BHD"],
         "context_filter": ""
        }
   ]
}


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
import random
import tflearn

words = []
classes = []
documents =[]
ignore_words = ['?']

# loop through each sentence in our intents pattern 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokanize each word in sentence
        w = nltk.word_tokenize(pattern)
        #add to our word list
        words.extend(w)
        # add to document in out corpus
        documents.append((w,intent['tag']))
        # add to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#remove duplicates
classes = sorted(list(set(classes)))

'''the above trasformation wont work on tensorflow so we need to 
transform further from documents of words to tensors of numbers'''

# create training data
training = []
output = []

# creat an empty array for output
output_empty = [0] * len(classes)

# training set , bag of words for each sentence 
for doc in documents:
    #initialize our bag of words
    bag = []
    # list of tokenized words for the pattern 
    pattern_words = doc[0]
    
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    # create our bag of words array 
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag,output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test list 
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 11)
net = tflearn.fully_connected(net, 11)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=20, show_metric=True)
model.save('model.tflearn')

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# load our saved model
model.load('./model.tflearn')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.60
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return random.choice(i['responses'])

            results.pop(0)
    else:
    	deff = "Sorry I did not get your question, Please try something else"
    	return deff


from flask import Flask, render_template, request
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)

#english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")

#english_bot.set_trainer(ChatterBotCorpusTrainer)
#english_bot.train("chatterbot.corpus.english")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    #return str(english_bot.get_response(userText))
    return response(userText)


if __name__ == "__main__":
    app.run()
