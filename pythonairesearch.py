# -*- coding: utf-8 -*-
"""
Created on Fri May 23 23:23:09 2025

@author: banern
"""
import math
import pandas as pd
import random
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
class MyBernoulliNB:
    def __init__(self):
        #We have two lists; words have been mapped to indicies, and each value in spam_word_probabilities or 
        #ham_word_probabilities is the probability that that word would appear in a message if it were spam or if it were not spam
        #our prior spam is the ratio of spam messages to all messages, and prior_ham is the ratio of regular messages to all messages
        self.spam_word_probabilities = []
        self.ham_word_probabilities = []
        self.prior_spam = 0.5
        self.prior_ham = 0.5
        self.num_spam = 0
        self.num_ham = 0
    def fit(self, train_vectors, train_labels):
        self.spam_word_probabilities = []
        self.ham_word_probabilities = []
        #WAIT REGARDING BELOW COMMENT- We may just want to assume equal likelihood to reduce bias in prediction
        #we need to know the prior; without any evidence, what is the probability of the message being spam?
        num_messages = len(train_vectors)
        for i in range(num_messages):
            if train_labels[i] == 1:
                self.num_spam += 1
        #go through all the labels, then determine the ratio of spam messages to total messages and normal messages to total messages
        self.num_ham = num_messages - self.num_spam
        """self.prior_spam = self.num_spam/num_messages
        self.prior_ham = 1 - self.prior_spam"""
        #iterate through all the indicies(words we mapped words to indicies earlier)
        #each word has an initial count of 0, where count is the total number of times the word appears in all spam messages
        for i in range(len(train_vectors[0])):
            count = 0
            for j in range(len(train_vectors)):
                if train_labels[j] == 1:
                    if train_vectors[j][i] == 1:
                        count += 1
                #if we see the word in this particular message, then increase the count so we can eventually calculate the probability that the word appears in a message given that the message is spam
            word_given_spam_prob = (count + 1)/(self.num_spam + 2)
            self.spam_word_probabilities.append(word_given_spam_prob)
        self.num_ham = num_messages - self.num_spam
        #now we do a similar double for loop for regular messages: given that this message is NOT spam, what is the probability that this particular word appears in the message?
        for i in range(len(train_vectors[0])):
            count = 0
            for j in range(len(train_vectors)):
                if train_labels[j] == 0:
                    if train_vectors[j][i] == 1:
                        count += 1
            word_given_ham_prob = (count + 1)/(self.num_ham + 2)
            self.ham_word_probabilities.append(word_given_ham_prob)
    def predict_one(self, message):
        #this function uses a naive version of bayes theorem (naive bayes model) 
        probability_x_given_spam = math.log(self.prior_spam)
        probability_x_given_ham = math.log(self.prior_ham)
        #for bayes theorem, we calculate: P(spam | x) = ((P(x | spam * P(spam))/ (P(x | spam) * P(spam) + P(x | ham) * P(ham)) ))
        for i in range(len(message)):
            if message[i] == 1:
                probability_x_given_spam += math.log(self.spam_word_probabilities[i])
            else:
                print(1 - self.spam_word_probabilities[i])
                probability_x_given_spam += math.log(1 - self.spam_word_probabilities[i])
        for i in range(len(message)):
            if message[i] == 1:
                probability_x_given_ham += math.log(self.ham_word_probabilities[i])
            else:
                probability_x_given_ham += math.log(1 - self.ham_word_probabilities[i])
        spam_bayes_numerator = probability_x_given_spam
        ham_bayes_numerator = probability_x_given_ham
        """total_spam = spam_bayes_numerator + math.log(self.prior_ham) + probability_x_given_ham
        spam_probability = spam_bayes_numerator/total_spam
        ham_probability = 1 - spam_probability
        if spam_probability > 0.5:
            return 1
        else:
            return 0"""
        return spam_bayes_numerator > ham_bayes_numerator
    def predict(self, all_messages):
        result = []
        for i in range(len(all_messages)):
            if(self.predict_one(all_messages[i])):
                result.append(1)
            else:
                result.append(0)
        return result
    def output_result(self, all_messages, vectors_to_messages, result):
        for i in range(len(all_messages)):
            if result[i] == 1: 
                print(vectors_to_messages[tuple(all_messages[i])] + ":  SPAM")
            else:
                print(vectors_to_messages[tuple(all_messages[i])] + ":  NOT SPAM")
    #returns a list of tuples, where each tuple is (value caluclated by ai, expected value)
    #the tuples are [((percentage calculated spam, actual percentage spam), (percentage calculated ham, actual percentage ham) (percentage accurate overall of ai, overall correctness will be 100% since its the original data))]
    def output_statistics(self, result, actual_results):
        list_of_stats = []
        confusion_matrix = []
        total_values = len(result)
        count_real_spam = 0
        count_real_ham = 0
        count_spam = 0
        count_ham = 0 
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        count_overall_accuracy = 0
        for i in range(total_values):
            if actual_results[i] == 1:
                count_real_spam += 1
            else:
                count_real_ham += 1
            if result[i] == 1:
                count_spam += 1
            elif result[i] == 0:
                count_ham += 1
            if result[i] == actual_results[i]:
                if result[i] == 1 and actual_results[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
                count_overall_accuracy += 1
            else:
                if result[i] == 0 and actual_results[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
        percent_spam_ai = (count_spam/total_values) * 100
        percent_ham_ai = (count_ham/total_values) * 100
        real_percent_spam = count_real_spam/total_values
        real_percent_ham = count_real_ham/total_values
        overall_accuracy_percentage = count_overall_accuracy/total_values
        actual_stats = []
        actual_stats.append(true_negatives)
        actual_stats.append(false_positives)
        confusion_matrix.append(actual_stats)
        actual_stats2 = []
        actual_stats2.append(false_negatives)
        actual_stats2.append(true_positives)
        confusion_matrix.append(actual_stats2)
        list_of_stats.append((count_spam, count_real_spam))
        list_of_stats.append((count_ham, count_real_ham))
        list_of_stats.append((percent_spam_ai, real_percent_spam))
        list_of_stats.append((percent_ham_ai, real_percent_ham))
        list_of_stats.append(confusion_matrix)
        return  list_of_stats
#we use the pandas library to create a dataframe object so we can parse and analyze the data
df = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
    sep = '\t',
    header = None,
    names = ['label', 'message']
    )
#we map the data to binary values, 0 if its a normal message, 1 if the message is spam 
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
#we need to parse the data; to do this, we need a few sets 
#all_words is a set containing all the words from all the emails in the data set
#messages_to_labels is a map that maps messages to their binary label(0 for normal message, 1 for spam message)
#messages_to_words is a dictionary of messages to all the words in that particular message
all_words = set()
messages_to_labels = dict()
messages_to_words = dict()
words_to_indicies = dict() 
count = 0
for i in range(5571):
    message = df['message'][i]
    messages_to_labels[message] = df['label'][i]
    messages_to_words[message] = set()
    word = ''
    for char in message:
        if char.isspace() or char == '.' or char == '?' or char == '!' or char == ',':
            word = word.lower()
            if word != '' and not word in all_words:
                all_words.add(word)
                messages_to_words[message].add(word)
                words_to_indicies[word] = count
                count += 1
            word = ''
            continue
        word = word + char
#we create the vectors mapped to the messages; we iterate through the messages to their words, then fill the 
messages_to_vectors = dict()
#once we train the model, we want to be able to output whether it is spam or not, we will do a bunch of shuffling, so we want to be able to trace the vectors back to the messages
vectors_to_messages = dict()
vectors_and_labels = []
for message in list(messages_to_words.keys()):
    newVec = []
    words = messages_to_words[message]
    count = 0
    for i in range(len(all_words)):
        newVec.append(0)
    for word in words:
        index = words_to_indicies[word]
        newVec[index] = 1
    messages_to_vectors[message] = newVec
    vectors_to_messages[tuple(newVec)] = message
    vectors_and_labels.append((newVec, messages_to_labels[message]))
"""list_of_vectors = []
list_of_labels = []
for message in list(messages_to_vectors.keys()):
    vector = messages_to_vectors[message]
    list_of_vectors.append(vector)
    list_of_labels.append(messages_to_labels[message])
for i in range(len(list_of_vectors)):
    vector = list_of_vectors[i]
    label = list_of_labels[i]
    vectors_and_labels.append((vector, label))"""
#shuffle the data in case the order is biased (if all the spam messages are at the beginning or at the end and otherwise we wouldn't be training our ai on any spam messages)
random.shuffle(vectors_and_labels)
shuffled_vectors = []
shuffled_labels = []
#we make the decision to use 90% of the data for training data and 10% of the data for testing data, then we create two new lists with the binary vectors and their corresponding labels in two different lists 
for i in range(len(vectors_and_labels)):
    shuffled_vectors.append(vectors_and_labels[i][0])
    shuffled_labels.append(vectors_and_labels[i][1])
separation = int(len(shuffled_vectors) * 0.9)
train_vectors = []
train_labels = []
test_vectors = []
test_labels = []
for i in range(separation):
    train_vectors.append(shuffled_vectors[i])
    train_labels.append(shuffled_labels[i])
for i in range(separation, len(shuffled_vectors)):
    test_vectors.append(shuffled_vectors[i])
    test_labels.append(shuffled_labels[i])
model = MyBernoulliNB()
model.fit(train_vectors, train_labels)
predicted_labels = model.predict(test_vectors)
statistics = model.output_statistics(predicted_labels, test_labels)
for statistic in statistics:
    print(statistic)

"""model = BernoulliNB()
model.fit(train_vectors, train_labels)
predicted_labels = model.predict(test_vectors)
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", accuracy)"""



    
    

    

    
    
    



       
        
        