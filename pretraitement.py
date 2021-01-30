# -*- coding: utf-8 -*-
"""
@author:Data Scientist
if you have any question My Email is:
gheffari.abdelfattah@gmail.com 

"""
import re
import codecs

     
def normalizeArabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return(text)

def remove_urls(text):
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+",'',text)
    return text

def remove_spec(text):
    text = re.sub('<.*?>+', '', text)
    return text

def remove_diacritics(text):
    regex = re.compile(r'[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]')
    return re.sub(regex, '', text)

def remove_numbers(text):
    regex = re.compile(r"(\d|[\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669])+")
    return re.sub(regex, ' ', text)

def remove_non_arabic_words(text):
    return ' '.join([word for word in text.split() if not re.findall(
        r'[^\s\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A]',
        word)])

def remove_extra_whitespace(text):
    text= re.sub(r'\s+', ' ', text)
    return re.sub(r"\s{2,}", " ", text).strip()

def remove_non_arabic_symbols(text):
    return re.sub(r'[^\u0600-\u06FF]', ' ', text)

def remove_repeating_char(text):
    return re.sub(r'(..)\1+', r'\1', text)

def pretraitement(tweet):
    tweet=remove_non_arabic_symbols(tweet)
    tweet=remove_repeating_char(tweet)
    tweet=remove_extra_whitespace(tweet)
    tweet=remove_non_arabic_words(tweet)
    tweet=remove_numbers(tweet)
    tweet=remove_spec(tweet)
    tweet=remove_urls(tweet)
    tweet=remove_diacritics(tweet)
    tweet=normalizeArabic(tweet)
    return tweet