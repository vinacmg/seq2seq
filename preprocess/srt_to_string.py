import pysrt

def list_subs(relative_path):
	try:
		subs = pysrt.open(relative_path)
	except:
		try:
			subs = pysrt.open(relative_path, encoding='iso-8859-1')
		except:
			print("char encoding issue")
			quit()

	slist = []

	for sub in subs:
		slist.append(sub.text.replace('\\n', '\n'))

	return slist

'''
slist = list_subs('../../../Subs/srt/Ex.Machina.2015.srt')
for sentence in slist[:10]:
	print(sentence)
'''