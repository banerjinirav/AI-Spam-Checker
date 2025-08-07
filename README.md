A Naive Bayes Spam Classifier built entirely from scratch in Python to detect spam vs. ham in SMS text messages.

What makes this project stand out is its custom noise reduction pipeline, which goes beyond basic preprocessing:

Absolute Frequency Filtering – drops low-frequency words with little predictive value.

Relative Probability Filtering – calculates the ratio of each word’s spam-vs-ham likelihood, removing neutral words that dilute classification accuracy.

Other features:

Laplace smoothing to handle unseen words.

Log-probability summation to avoid numerical underflow.

Confusion matrix analysis to identify and debug misclassifications.

This project demonstrates my ability to design, debug, and optimize NLP systems from first principles — a skill set that applies directly to real-world AI reliability challenges.
