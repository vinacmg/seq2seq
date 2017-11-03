from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Text, FreqDist
from srt_to_string import list_subs
import json
import glob


def replace_all(string, proib_list, new_word=''):
	for word in proib_list:
		string = string.replace(word, new_word)

	return string


def build_dictionaries(tokens, vocab_size):

	#considerando EOS e PAD dicionario tem vocab_size + 2 de tamanho
	num2word = {}
	word2num = {}

	fdist = FreqDist(tokens)
	most_common = fdist.most_common(vocab_size)
	sort = sorted(most_common)
	
	for i in range(0, vocab_size):
		num2word[i+2] = sort[i][0]
		word2num[sort[i][0]] = i+2

	num2word[0] = "PAD"
	num2word[1] = "EOS"
	word2num[0] = "PAD"
	word2num[1] = "EOS"

	return num2word, word2num

def save_dictionaries(num2word, word2num):

	with open('num2word.txt', 'w') as f:

		json.dump(num2word, f)

	with open('word2num.txt', 'w') as f:

		json.dump(word2num, f)

def load_dictionaries():

	with open('num2word.txt', 'r') as f:

		num2word = json.load(f)

	for i in range(0, len(num2word)):
		num2word[i] = num2word[str(i)]
		del num2word[str(i)]

	with open('word2num.txt', 'r') as f:

		word2num = json.load(f)

	return num2word, word2num


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
	
	sentences_talked += (sentences_all[:-1])
	sentences_answered += (sentences_all[1:])
	tokens += (tkns)


print(sentences_talked)



#tokens_set = set(tokens)

'''
numword, wordnum = build_dictionaries(tokens, 50)
save_dictionaries(numword, wordnum)

num2word, word2num = load_dictionaries()

print(word2num)
print(numword)'''