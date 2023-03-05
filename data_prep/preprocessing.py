import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

import contractions
from spellchecker import SpellChecker

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt') 
nltk.download('stopwords')

# initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Create a spellchecker object for English
spell = SpellChecker(language='en')

warnings.filterwarnings("ignore")


contractions.contractions_dict['dont'] = 'do not'
contractions.contractions_dict['didnt'] = 'did not'
contractions.contractions_dict['couldnt'] = 'could not'
contractions.contractions_dict['cant'] = 'can not'
contractions.contractions_dict['doesnt'] = 'does not'
contractions.contractions_dict['wont'] = 'would not'
contractions.contractions_dict['shouldnt'] = 'should not'


def text_cleaning(text):
    # creating an empty list
    expanded_words = []

    # Perform contractions to convert words like don't to do not
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)

    # tokenizing text
    tokens = word_tokenize(expanded_text)

    # converting list to string
    text = ' '.join(tokens)

    # convert text to lowercase and remove leading/trailing white space
    text = ''.join(text.lower().strip())

    # remove newlines, tabs, and extra white spaces
    text = re.sub('\n|\r|\t', ' ', text)
    text = re.sub(' +', ' ', text)
    text = ''.join(text.strip())

    # remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join([word for word in text.split() if word not in stop_words])
    cleaned_text = ''.join([char for char in cleaned_text if char not in string.punctuation])
    cleaned_text = ' '.join(
        [char for char in cleaned_text.split() if len(char) > 2])  # Added this for only keeping words with lengths>2

    return cleaned_text


def get_corrections(df):
    
    for i in range(len(df)):
        corrected_words = []

        tokenized_split = df.cleaned_tokenize_text[i].split()
        corrected_split = df.corrected_text[i].split()
        mistakes_split = df.mistakes[i].split()

        assert len(tokenized_split) == len(corrected_split)

        for j, c in enumerate(tokenized_split):
            if c in mistakes_split:
                corrected_words.append(corrected_split[j])

        df.loc[i, "corrected_words"] = " ".join(corrected_words)
    
    return df


def lemmatize_with_pos(word):
    pos = get_wordnet_pos(word)
    if pos:
        return lemmatizer.lemmatize(word, pos=pos)
    else:
        return lemmatizer.lemmatize(word)

# define a function to get the appropriate POS tag for a word
def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # default to noun if not found

# define a function to apply lemmatization to each word
def lemmatize_text(text):
    return [lemmatize_with_pos(word) for word in text]


# Define a function to count the number of spelling mistakes in a given essay
def count_spelling_mistakes(df):
    for j in range(len(df)):
        essay = df.cleaned_tokenize_text[j].split()
        
        misspelled = list(spell.unknown(essay))
        
        mistakes, corrections = [], []
        for i, m in enumerate(misspelled):
            s = spell.correction(m)
            if s:
                corrections.append(s)
                mistakes.append(m)

        for i, m in enumerate(mistakes):
            indexes = [i for i, j in enumerate(essay) if j == m]
            for ind in indexes:
                essay[ind] = corrections[i]

        essay = " ".join(essay)
        n_mistakes = len(mistakes)
        mistakes = " ".join(mistakes)
        corrections = " ".join(corrections)

        df.loc[j, ['corrected_text', 'mistakes', 'corrected_words', 'num_mistakes']] = [essay, mistakes, corrections, n_mistakes]
    return df


def count_pos_tags(tokens):
    noun_count = 0
    verb_count = 0
    adjective_count = 0
    adverb_count = 0

    # loop through each token and increment the corresponding counter
    for token, tag in nltk.pos_tag(tokens):
        if tag.startswith('N'):  # noun
            noun_count += 1
        elif tag.startswith('V'):  # verb
            verb_count += 1
        elif tag.startswith('J'):  # adjective
            adjective_count += 1
        elif tag.startswith('R'):  # adverb
            adverb_count += 1

    # return a dictionary with the counts
    return {'noun': noun_count, 'verb': verb_count, 'adjective': adjective_count, 'adverb': adverb_count}


# define a function to assign score category based on scores
def assign_score_category(row):
    if (row[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']] <= 2.5).sum() > 4:
        return 'low'
    elif (row[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']] >= 4).sum() > 4:
        return 'high'
    else:
        return 'medium'


def list_to_string(lst):
    return ' '.join(lst)


def process_df(df):

    # 1. apply the text_cleaning function to the 'full_text' column using apply() method
    df['cleaned_tokenize_text'] = df['full_text'].apply(text_cleaning)

    # 2. Apply the function to the tokenized text column and store the results in new columns
    df = count_spelling_mistakes(df)
    
    # 3. apply lemmatize_text function to the corrected_text
    df['lemmatized_text'] = df['corrected_text'].apply(lambda x: lemmatize_text(x.split()))
    
    # 4. Compute the statistics
    df['sent_count'] = df['full_text'].apply(lambda x: len(sent_tokenize(x)))

    # 5. Compute the average number of words in a sentence in an essay
    df['sent_len'] = df['full_text'].apply(lambda x: np.mean([len(w.split()) for w in sent_tokenize(x)]))

    # 6. Apply the count_pos_tags function to each row
    df['pos_counts'] = df['lemmatized_text'].apply(count_pos_tags)
    
    # 7. Compute the word count for each essay
    df['word_count'] = df.full_text.apply(lambda x: len(x.split()))

    # 8. Extract the count for each POS tag into a separate column
    df['noun_count'] = df['pos_counts'].apply(lambda x: x['noun'])
    df['verb_count'] = df['pos_counts'].apply(lambda x: x['verb'])
    df['adjective_count'] = df['pos_counts'].apply(lambda x: x['adjective'])
    df['adverb_count'] = df['pos_counts'].apply(lambda x: x['adverb'])

    # 9. apply the function to create a new column
    if "cohesion" in df.columns:
        df['score_category'] = df.apply(assign_score_category, axis=1)

    # 10. drop the tokens and pos_counts columns
    df = df.drop(['pos_counts'], axis=1)
    
    df['lemmatized_text'] = df['lemmatized_text'].apply(list_to_string)
    df['num_mistakes'] = df['num_mistakes'].apply(int)
    df['sent_len'] = df['sent_len'].apply(int)
    df['sent_count'] = df['sent_count'].apply(int)

    return df


if __name__=='__main__':
    df = pd.read_csv('../data/train.csv')
    df = process_df(df)
    df.to_csv("../data/processed_essays.csv", index=False)
    print(df.head())
