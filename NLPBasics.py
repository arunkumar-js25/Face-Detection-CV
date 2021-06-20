
#STRING MANIPULATION
string = "GURUGRAM"
print(string[0])                            #get first alphabet with index                  #output: G
print(string[2], string[5])                 #printing multiple alphabets                    #output: RR
print(string[-4])                           #for getting alphabet with negative indexing    #output: G

#STRIP
sentence = "****Hello World! I am Amit Chauhan****"
removing_character = "*"
sentence.strip(removing_character)          #using strip function to remove star(*)         #output: 'Hello World! I am Amit Chauhan'

#STR JOIN
str1 = "Happy"
str2 = "Home"
" Good ".join([str1, str2])                                                                 #output: 'Happy Good Home'

#REGEX
import re
sentence = "My computer gives a very good performance in a very short time."
string = "very"
str_match = re.search(string, sentence)
print(str_match)                                                                            #output: <re.Match object; span=(20, 24), match='very'>
str_match.span()
find_all = re.findall("very", sentence)
for word in re.finditer("very", sentence):
    print(word.span())

#Tokenization - When a sentence breakup into small individual words, these pieces of words are known as tokens, and the process is known as tokenization.
#Stemming - Stemming is a process in which words are reduced to their root meaning.
#Types of stemmer -->   Porter Stemmer      --> It is used for the reduction of a word to its stem or root word.
#                       Snowball Stemmer    --> Snowball stemmer is used for a more improved method
#Lemmatization - Lemmatization is better than stemming and informative to find beyond the word to its stem also determine part of speech around a word. That’s why spacy has lemmatization, not stemming
#Stop word is used to filter some words which are repeat often and not giving information about the text
#Part of Speech (POS) - Part of speech is a process to get information about the text and words as tokens, or we can say grammatical information of words.
    #There are two types of tags. For the noun, verb coarse tags are used, and for a plural noun, past tense type, we used fine-grained tags. So the coarse tag is a NOUN, and the fine grain tag is NN
#Named Entity Recognition (NER) - Named entity recognition is very useful to identify and give a tag entity to the text, whether it is in raw form or in an unstructured form. Sometimes readers don't know the type of entity of the text so, NER helps to tag them and give meaning to the text.

import spacy
load_en = spacy.load('en_core_web_sm')

#TOKENIZATION
words = load_en("I'm going to meet\ M.S. Dhoni.")
for tokens in words:
    print(tokens.text)

#PORTER STEMMER
from nltk.stem.porter import PorterStemmer
pot_stem = PorterStemmer()
words = ['happy', 'happier', 'happiest', 'happiness', 'breathing', 'fairly']
for word in words:
    print(word + '----->' + pot_stem.stem(word))

#SNOWBALL STEMMER
from nltk.stem.snowball import SnowballStemmer
snow_stem = SnowballStemmer(language='english')
for word in words:
    print(word + '----->' + snow_stem.stem(word))

#LEMMATIZATION
example_string = load_en(u"I'm happy in this happiest place with all happiness. It feels how happier we are")
for lem_word in example_string:
    print(lem_word.text, '\t', lem_word.pos_, '\t', lem_word.lemma, '\t', lem_word.lemma_)

#STOP WORDS
print(load_en.Defaults.stop_words)

#POS
str1 = load_en(u"This laptop belongs to Amit Chauhan")
print(str1[1].pos_)                             #pos_ tag operation                     #output: NOUN
print(str1[1].tag_)                             #to know fine grained information       #output: NN

pos_count = str1.count_by(spacy.attrs.POS)
print(pos_count)                                #output: {90: 1, 92: 1, 100: 1, 85: 1, 96: 2}
print(str1.vocab[90].text)                             #output: DET

#NER
doc = load_en(u" I am living in India, Studying in IIT")                             #lets label the entity in the text file
if doc.ents:
    for ner in doc.ents:
        print(ner.text + ' - '+ ner.label_ + ' - ' + str(spacy.explain(ner.label_))) #output: India - GPE - Countries, cities, states
else:
    print("No Entity Found")
