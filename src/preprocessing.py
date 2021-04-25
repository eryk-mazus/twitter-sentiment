
# imports
import pandas as pd
import re
import logging

# inits
logging.basicConfig(level=logging.INFO)

# read slang/emoticons dictionaries
emo_dict = pd.read_excel('src/dictionaries/emoticons.xlsx')
emo_dict = emo_dict.set_index('emoticon')['description'].to_dict()

slang_dict = pd.read_excel('src/dictionaries/slang.xlsx')
slang_dict = slang_dict.set_index('slang')['meaning'].to_dict()

def copy_df(df):
    return df.copy()


def remove_retweet_info(df, column):  
    logging.info(' > removing retweet info...')  
    df[column] = df[column].str.replace(r'(rt|RT)(?:\b\W*@(\w+))+:', '', regex=True)
    return df


def replace_urls(df, column):
    """eg: https://t.co/1bcCdL1csR ==> ''"""
    logging.info(' > replacing urls with empty space...') 
    # df[column] = df[column].str.replace(r'https?:\/\/\S*', 'url', flags=re.MULTILINE, regex=True)
    df[column] = df[column].str.replace(r'https?:\/\/\S*', '', flags=re.MULTILINE, regex=True)
    return df


def remove_hashtags(df, column):
    """remove hashtags in front of words"""
    logging.info(' > removing hashtags in front of the words...') 
    df[column] = df[column].str.replace(r'#([^\s]+)', r'\1', flags=re.MULTILINE, regex=True)
    return df


def remove_targets(df, column):
    """for now, remove @user with empty string"""
    logging.info(' > removing @targets...') 
    df[column] = df[column].str.replace(r'@[^\s]+', '', flags=re.MULTILINE, regex=True)
    return df


def remove_new_line_chars(df, column):
    logging.info(' > removing new line characters...') 
    df[column] = df[column].str.replace("\n", '', flags=re.MULTILINE, regex=False)
    return df


def replace_abbreviations(df, column):
    # todo: add trange to visualize progress
    logging.info(' > replacing abbreviations...')
    df = df.copy()
    for abb, meaning in slang_dict.items():
        if isinstance(abb, int):
            abb = str(abb)
        regex = re.compile(r"\b({})\b".format(abb))
        df[column] = df[column].str.replace(regex, meaning, regex=True)
    return df


def replace_emoticons(df, column):
    # todo: add trange to visualize progress
    logging.info(' > replacing emoticons...')
    df = df.copy()
    for emo, meaning in emo_dict.items():
        regex = re.compile(r"\b({})\b".format(re.escape(emo)))
        df[column] = df[column].str.replace(regex, meaning, regex=True)
    return df


def replace_consecutive_chars(df, column):
    logging.info(' > replacing three+ consecutive characters with two...')
    # https://stackoverflow.com/questions/10072744/remove-repeating-characters-from-words
    df[column] = df[column].str.replace(r"(.)\1{2,}", r"\1\1", regex=True)
    return df


def preprocess_column(df, tweets_col):
    df_pp = (df.pipe(remove_retweet_info, tweets_col)
               .pipe(replace_urls, tweets_col)
               .pipe(remove_hashtags, tweets_col)
               .pipe(remove_targets, tweets_col)
               .pipe(remove_new_line_chars, tweets_col)
            #    .pipe(replace_emoticons, tweets_col)
               .pipe(replace_abbreviations, tweets_col)
               .pipe(replace_consecutive_chars, tweets_col))
    return df_pp


if __name__ == "__main__":
    pass

