from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Text, FreqDist
import json
import glob
import copy

try:
	from .srt_to_string import list_subs
except: 
	from srt_to_string import list_subs


def replace_all(string, proib_list, new_word=''):
	for word in proib_list:
		string = string.replace(word, new_word)

	return string


def build_dictionaries(tokens, vocab_size):

	num2word = {}
	word2num = {}

	fdist = FreqDist(tokens)
	most_common = fdist.most_common(vocab_size)
	sort = sorted(most_common)
	
	for i in range(3, vocab_size):
		num2word[i] = sort[i-3][0]
		word2num[sort[i-3][0]] = i

	num2word[0] = "PAD"
	num2word[1] = "EOS"
	num2word[2] = "UNK"
	word2num["PAD"] = 0
	word2num["EOS"] = 1
	word2num["UNK"] = 2

	return num2word, word2num

def save_dictionaries(num2word, word2num):

	with open('num2word.txt', 'w') as f:

		json.dump(num2word, f)

	with open('word2num.txt', 'w') as f:

		json.dump(word2num, f)

def load_dictionaries(directory):

	with open(directory + 'num2word.txt', 'r') as f:

		num2word = json.load(f)

	for i in range(0, len(num2word)):
		num2word[i] = num2word[str(i)]
		del num2word[str(i)]

	with open(directory + 'word2num.txt', 'r') as f:

		word2num = json.load(f)

	return num2word, word2num


def save_sentences(sentences_talked, sentences_answered, tokens):

	with open('sentences_talked.txt', 'w') as f:

		json.dump(sentences_talked, f)

	with open('sentences_answered.txt', 'w') as f:

		json.dump(sentences_answered, f)

	with open('tokens.txt', 'w') as f:

		json.dump(tokens, f)

def load_sentences(directory):

	with open(directory + 'sentences_talked.txt', 'r') as f:

		sentences_talked = json.load(f)

	with open(directory + 'sentences_answered.txt', 'r') as f:

		sentences_answered = json.load(f)

	with open(directory + 'tokens.txt', 'r') as f:

		tokens = json.load(f)

	return sentences_talked, sentences_answered, tokens





dir_list = glob.glob('../../../Subs/srt/*.srt')

tags = ['<i>','</i>','{i}','{/i}','<b>','</b>','{b}','{/b}','<u>','</u>','{u}','{/u}','\"','\''] #problemas com aspas


def process_srt(directory):

	sublist = list_subs(directory)

	no_tags_sents = [replace_all(sent, tags) for sent in sublist]
	lower_sents = [sent.lower() for sent in no_tags_sents]

	sentences = [word_tokenize(sent) for sent in lower_sents]
	sentences = sentences[3:-3] #retirar possíveis créditos da legenda

	tokens = []

	for i in sentences:
		for j in i:
			tokens.append(j)

	return sentences, tokens


sentences_talked = []
sentences_answered = []
tokens = []

for srt in dir_list:

	sentences_all, tkns = process_srt(srt)
	
	sentences_talked += copy.deepcopy(sentences_all[:-1]) #copy values. not references
	sentences_answered += copy.deepcopy(sentences_all[1:])
	tokens += (tkns)
'''
num2word, word2num = build_dictionaries(tokens, 10)
save_dictionaries(num2word ,word2num)'''
num2word, word2num = load_dictionaries("")


for sent in sentences_talked:
	for i in range(0, len(sent)):
		try:
			sent[i] = word2num[sent[i]]
		except:
			sent[i] = word2num['UNK']

for sent in sentences_answered:
	for i in range(0, len(sent)):
		try:
			sent[i] = word2num[sent[i]]
		except:
			sent[i] = word2num['UNK']
