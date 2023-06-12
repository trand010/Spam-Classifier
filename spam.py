import math
import os
import numpy as np
from collections import Counter



# Calculates the log of the terms as per normal distribution
def calcLogGausProbability(x, mean, stdDev):
    exponent = (x - mean) ** 2 / (2 * stdDev ** 2)
    coefficient = -0.5 * math.log(2 * math.pi * stdDev ** 2)
    return coefficient + (-exponent), exponent

# Predicts the class of the function
def classify(featSummary, inputInstance):
    probabilities = {}
    emailCount = TOTAL_EMAIL_COUNT
    
    for classLabel, classSummary in featSummary.items():
        logTerms = expoTerms = 0
        for i, (mean, stdDev) in enumerate(classSummary):
            x = inputInstance[i]
            if stdDev:
                logTerm, expTerm = calcLogGausProbability(x, mean, stdDev)
                logTerms += logTerm
                expoTerms += expTerm
        probabilities[str(classLabel)] = logTerms - expoTerms + math.log((classCount[classLabel] / emailCount))

    finalLabel = None
    maxProb = float("-inf")
    for label, prob in probabilities.items():
        if prob > maxProb:
            finalLabel = float(label)
            maxProb = prob
    return finalLabel

def standDeviation(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# Calculates the summary of features for each class.
def sumLabels(emailFeatMatrix, emailLabels):
    global classCount
    classCount = {}
    featSummary = {}
    for label in emailLabels:
        classCount[label] = classCount.get(label, 0) + 1
    
    partitioned = {}
    for features, label in zip(emailFeatMatrix, emailLabels):
        if label not in partitioned:
            partitioned[label] = []
        partitioned[label].append(features)
    for classLabel, instances in partitioned.items():
        featSummary[classLabel] = [(sum(attribute) / len(attribute), standDeviation(attribute)) for attribute in zip(*instances)]
    return featSummary

def getClassifi(featSummary, inputInstances):
    classifications = []
    for instance in inputInstances:
        classification = classify(featSummary, instance)
        classifications.append(classification)
    return classifications

def accuracy(testLabels, prediLabels):
    correctPredi = sum(t == p for t, p in zip(testLabels, prediLabels))
    score = (correctPredi / len(testLabels) * 100)
    print("Out of", len(testLabels), "test emails,", correctPredi, "were properly identified!")

    print(f"Accuracy is {score:.2f}%")

# Reads all emails and creates a list of tuples of the 3000 most common words.
def createWordDB(directory: str):
    emails = [os.path.join(directory, f) for f in os.listdir(directory)]
    allWords = []
    for mail in emails:
        with open(mail) as f:
            allWords.extend(f.read().split())
    wordCounts = Counter(allWords)
    for word in list(wordCounts.keys()):
        if not word.isalpha() or len(word) == 1:
            del wordCounts[word]
    topWords = wordCounts.most_common(3000)
    topWordID = {word: index + 1 for index, (word, count) in enumerate(topWords)}
    mailCount = len(emails)
    global TOTAL_EMAIL_COUNT
    TOTAL_EMAIL_COUNT = mailCount
    return topWordID

# Creates a tabular representation of the dataset.
def extractFeats(directory, topWordID):
    emailFiles = [os.path.join(directory, fi) for fi in os.listdir(directory)]
    emailFeatMatrix = np.zeros((len(emailFiles), 3000))
    instance_labels = np.zeros(len(emailFiles))
    for mail_id, mail in enumerate(emailFiles):
        
        with open(mail) as f:
            for num, line in enumerate(f):
                if num == 2:
                    words = line.split()
                    for word in words:
                        if word in topWordID:
                            emailFeatMatrix[mail_id, topWordID[word] - 1] += words.count(word)

        filename = os.path.basename(mail)
        if filename.startswith("spmsg"):
            instance_labels[mail_id] = 0
        else:
            instance_labels[mail_id] = 1
    return emailFeatMatrix, instance_labels

def naiveBayesClassi(train_dir="train-mails", test_dir="test-mails"):
    # Create word database
    print("Creating word database...\n")
    topWordID = createWordDB(train_dir)
    # Extract features from training data
    train_features_matrix, train_email_labels = extractFeats(train_dir, topWordID)
    # Extract features from test data
    test_feature_matrix, testLabels = extractFeats(test_dir, topWordID)
    # Train model on training data
    print("Training model...\n")
    model = sumLabels(train_features_matrix, train_email_labels)
    # Make predictions on test data
    print("Making predictions on test data...\n")
    prediLabels = getClassifi(model, test_feature_matrix)
    # Calculate accuracy score
    accuracy(testLabels, prediLabels)
    
    