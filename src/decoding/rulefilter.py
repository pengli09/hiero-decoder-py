#-*- coding: utf-8 -*-
'''
Created on Apr 30, 2014

@author: lpeng
'''
import argparse
import sys
from ioutil import Reader, Writer

def get_all_possible_src_str(sentences, max_X_len):
  src_strs = set()
  with Reader(sentences) as sentences:
    for sentence in sentences:
      sentence = sentence.strip().split(' ')
      sen_len = len(sentence)
      for width in range(1, max_X_len+1):
        for start in range(0, sen_len-width+1):
          # init phrase translations
          src_phrase = ' '.join(sentence[start:start+width])
          src_strs.add(src_phrase)
          
          # one nonterminal
          for x_start in range(start, start+width-1+1):
            for x_end in range(x_start+1, start+width+1):
              if x_end-x_start == width:
                # at least one word should be left
                continue
            
              words = sentence[start:x_start]
              words.append('|')
              words.extend(sentence[x_end:start+width])
              src_phrase = ' '.join(words)
              src_strs.add(src_phrase)
              
          # two nonterminals
          for x1_start in range(start, start+width-3+1):
            for x1_end in range(x1_start+1, start+width-2+1):
              # there should be at least one word between X1 and X2
              for x2_start in range(x1_end+1, start+width-1+1):
                for x2_end in range(x2_start+1, start+width+1):
                  words = sentence[start:x1_start]
                  words.append('|')
                  words.extend(sentence[x1_end:x2_start])
                  words.append('|')
                  words.extend(sentence[x2_end:start+width])
                  src_phrase = ' '.join(words)
                  src_strs.add(src_phrase)
      
  return src_strs

def filter_rule_table(sentences, rule_table, output_file, max_X_len):
  print >> sys.stderr, 'find all possible source strings....'
  src_strs = get_all_possible_src_str(sentences, max_X_len)
  print >> sys.stderr, '%d distinct strings found' % len(src_strs)
  
  print >> sys.stderr, 'filtering rule table....'
  rule_id = 0
  with Reader(rule_table) as rule_table:
    with Writer(output_file) as writer:
      for rule in rule_table:
        src = rule[0:rule.find(' ||| ')].replace('|0', '|').replace('|1', '|')
        if src in src_strs:
          writer.write(rule.strip() + ' ||| ' + str(rule_id) + '\n')
        rule_id += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('sentences')
  parser.add_argument('rule_table')
  parser.add_argument('output_file')
  parser.add_argument('max_X_len', type=int)
  options = parser.parse_args()
  
  filter_rule_table(options.sentences, options.rule_table, options.output_file,
                    options.max_X_len)
  print >> sys.stderr, 'Done!'