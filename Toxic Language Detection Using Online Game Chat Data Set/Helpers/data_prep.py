import pandas as pd
import numpy as np
import nltk


# Deletes columns in dataset
def delete_columns(dataset, column_name_list):
    dataset.drop(column_name_list, axis=1, inplace=True)
    return dataset


def create_column(dataset, column_name):
    dataset[column_name] = np.nan


def lower_letters(dataset, column_name):
    dataset[column_name] = dataset[column_name].str.lower()  # lowers uppercase letters


def stop_words_creator(lang):
    from nltk.corpus import stopwords
    stop_words = stopwords.words(lang)
    stop_words_df = pd.DataFrame({'words': stop_words})  # creates a DF that contains a column named 'words'
    stop_words = pd.DataFrame({'words': stop_words})  # Converts stop_words list to a DF
    stop_words_df['words'] = stop_words_df['words'].str.replace('[^\w\s]', '')  # Removes punc.
    stop_words = stop_words.append(stop_words_df, ignore_index=True)  # Appends DFs
    return stop_words['words'].values.tolist()


def remove_stop_words(dataset, column_name, stop_words_to_remove):
    dataset[column_name] = dataset[column_name].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in stop_words_to_remove))


def remove_numbers(dataset, column_name):
    dataset[column_name] = dataset[column_name].str.replace('\d', '')

def remove_punctuations(dataset, column_name):
    dataset[column_name] = dataset[column_name].str.replace('[^\w\s]', '')


def remove_rare_words(dataset, column_name, freq=5):
    temp_df = pd.Series(' '.join(dataset[column_name]).split()).value_counts()
    rare_words = temp_df[temp_df <= freq]
    dataset[column_name] = dataset[column_name].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in rare_words))


def remove_champ_names(dataset, column_name, champ_list='champion_name'):
    raw_champ_names = dataset[champ_list].tolist()
    champ_names = list(dict.fromkeys(raw_champ_names))
    dataset[column_name] = dataset[column_name].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in champ_names))


def remove_gaming_acronyms(dataset, column_name):
    gaming_acronyms = ['gg', 'pls', 'lag', 'afk', 'aoe', 'brb', 'dps', 'dot', 'ff', 'blue', 'red', 'cd', 'rq', 'ggwp',
                       'ggg', 'gj', 'ty', 'thx', 'ss', 'np', 'wp', 'bg', 'man', 'ez', 'easy', 'premade', 'smiteless',
                       'smite', 'ok', 'report', 'ward', 'cmon', 'dude', 'rofl', 'lmao', 'hi', 'premades', 'omg', 'lol',
                       'nvm', 'bot', 'mid', 'top', 'jungle', 'jgl']
    dataset[column_name] = dataset[column_name].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in gaming_acronyms))


def remove_na_msgs(dataset, column_name):
    dataset.drop(dataset[dataset[column_name].map(len) == 0].index, inplace=True)


def detect_msgs(dataset, column_name, lang='en'):
    def detect_language(text):
        from langdetect import detect
        try:
            return detect(text)
        except:
            return 'unknown'

    dataset['msg_lang'] = dataset[column_name].apply(detect_language)
    dataset.drop(dataset[dataset['msg_lang'] != lang].index, inplace=True)


def df_tokenization(dataset, column_name):
    from textblob import TextBlob
    dataset[column_name].apply(lambda x: TextBlob(x).words).head()


def shape_dataframe(dataset, column_name, msg_len):
    dataset.drop(dataset[dataset[column_name].map(len) < msg_len].index, inplace=True)


def df_lemmatization(dataset, column_name):
    from textblob import Word
    dataset[column_name] = dataset[column_name].apply(
        lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


def sentiment_analysis(dataset, compound_score_column, column_name):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    dataset[compound_score_column] = dataset[column_name].apply(lambda x: sia.polarity_scores(x)["compound"])


def remove_pos_compound_scores(dataset, column='compound_score', target_compound_score=0):
    dataset.drop(dataset[dataset[column] >= target_compound_score].index, inplace=True)


def correct_words(dataset, column_name):
    from textblob import TextBlob
    dataset[column_name] = [str(TextBlob(word).correct()) for word in dataset[column_name]]


def quantile_func(dataset, target_dataset, column_name, target_column, msg_len, q1_perc=0.25, q3_perc=0.75):
    dataset[column_name] = dataset[column_name].astype(str)
    dataset[msg_len] = dataset[column_name].map(len)
    q1 = dataset[msg_len].quantile(q1_perc)
    q3 = dataset[msg_len].quantile(q3_perc)
    iqr = q3 - q1
    up_val = q3 + 1.5 * iqr
    low_val = q1 - 1.5 * iqr
    target_dataset[target_column] = dataset[column_name][
        dataset[column_name][(dataset[msg_len] < low_val) | (dataset[msg_len] > up_val)]]
