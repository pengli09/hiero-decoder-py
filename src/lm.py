import srilm
import sys
import time
import math
import logging

logger = logging.getLogger(__name__)

class LanguageModel:
	'''Language model.'''

	LOG10 = math.log(10)

	def __init__(self, filename, order):
		logger.info('Read language model file "%s", ' % filename)
		tb = time.time()
		self.vocab = srilm.Vocab()
		self.ngram = srilm.Ngram(self.vocab, order)
		self.ngram.read(filename)
		self.order = order
		te = time.time()
		logger.info('Language model loaded in ' + str(te - tb) + ' second(s)')

	def get_prob(self, words, begin=0, end=-1, context_start=0, verbose=0):
		return self.LOG10*self.get_prob_log10(words, begin, end, context_start,
																				  verbose)

	def get_prob_log10(self, words, begin=0, end=-1, context_start=0, verbose=0):
		order = self.order
		'''Get the probability'''
		prob = 0.0
		if end == -1:
			end = len(words)
		word_idxs = [self.vocab.index(w) for w in words]
		for i in xrange(begin, end):
			p = 0.0
			context_i = max(context_start, i - order + 1)
			p = self.ngram.wordprob(word_idxs[i], word_idxs[context_i:i])
			if p < -99:
				p = -99  # in case for -inf
			if verbose:
				fmt = "p('%s' | '%s') =  %13.11f"
				print fmt % (words[i], ' '.join(words[context_i:i]), p)
			prob += p
		return prob

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	lm = LanguageModel('/Users/lpeng/git-others/psrilm/english_30K_low_kn_o4_130205.gz', 4)
	print lm.get_prob('<s> president will in april london visit </s>'.split(), 1, -1, verbose=1)
	print lm.get_prob('<s> bush held a talk with sharon </s>'.split(), 5, 8, verbose=1)
	while True:
		sen = sys.stdin.readline()
		if len(sen) == 0 or sen == 'EXIT':
			break
		print lm.get_prob(sen.split(), 0)