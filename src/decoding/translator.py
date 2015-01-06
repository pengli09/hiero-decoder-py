#-*- coding: utf-8 -*-
'''
Created on Apr 26, 2014

@author: lpeng
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import re
import argparse
import signal
from multiprocessing import Pool
import logging.config
import json

from decoding.recombination import CombinedRecombinationChecker
from decoding.ckydecoder import CKYDecoder
from decoding.config import Config
import decoding.rules as rules 
from decoding.rules import RuleTable
from lm import LanguageModel
from timeutil import Timer
from ioutil import Reader, Writer
  
def output_translation(writer, translation, sid, 
                       output_features, with_rule_tree):
  parts = [str(sid)]
  parts.append(translation.translation)
  if output_features:
    parts.append(' '.join([str(f) for f in translation.features]))
    parts.append(str(translation.score))
  if with_rule_tree:
    parts.append(str(translation.rule_tree))
  writer.write(' ||| '.join(parts))
  writer.write('\n')  
 
  
def init_worker():
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  
 
def build_extra_feature_funcs(config):
  extra_feat_funcs = []
  # register new features here
  return extra_feat_funcs
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', required=True)
  parser.add_argument('-k', '--kbest', type=int, default=1)
  parser.add_argument('--drop-oov', action='store_true')
  parser.add_argument('-d', '--debug', action='store_true')
  parser.add_argument('-f', '--features', action='store_true')
  parser.add_argument('-i', '--input', default='-')
  parser.add_argument('-o', '--output', default='-')
  parser.add_argument('--checking', action='store_true')
  parser.add_argument('--expend-loser', action='store_true')
  parser.add_argument('--with-rule-tree', action='store_true')
  parser.add_argument('-t', '--threads', default='1', type=int)
  parser.add_argument('-l', '--logger-config', default=None)
  options = parser.parse_args()
  
  logger = logging.getLogger()

  logger_config = options.logger_config
  if logger_config is None:
    logging.basicConfig(level=logging.INFO)
  else:
    with open(logger_config) as logger_config:
      config_ = json.load(logger_config)
    logging.config.dictConfig(config_)
  
  k = options.kbest
  drop_oov = options.drop_oov
  debug = options.debug
  output_features = options.features
  checking = options.checking
  expend_loser = options.expend_loser
  with_rule_tree = options.with_rule_tree
  threads = options.threads
  logger.info('process num: %d' % threads)
   
  if options.input == '-':
    source = sys.stdin # TODO encoding
  else:
    source = Reader(options.input)
    
  if options.output == '-':
    writer = sys.stdout
  else:
    writer = Writer(options.output)
    
  if debug:
    rules.DEBUG = 1
  
  config = Config(options.config)
  if logger.level <= logging.INFO:
    config.write(sys.stderr)
    
  lm = LanguageModel(config.lm_file, config.lm_order)
  rule_table = RuleTable.load(config.rule_table_file, lm, config)
  
  extra_feature_funcs = build_extra_feature_funcs(config)
  recombination_checker = CombinedRecombinationChecker(extra_feature_funcs)
  decoder = CKYDecoder(config, rule_table, lm, 
                       recombination_checker=recombination_checker,
                       extra_feature_funcs=extra_feature_funcs,
                       checking_hypo=checking, expend_loser=expend_loser)
  
  logger.info('Start decoding...')
  def translate(data):
    _, sentence = data
    translations = decoder.translate(sentence, k, drop_oov, with_rule_tree)
    
    return translations
  
  total_timer = Timer()
  total_timer.tic()
  timer = Timer()
  sentences = [re.sub(r'\s+', ' ', sentence).strip().split(' ') for
               sentence in source]
  data = [(sid, sentences[sid]) for sid in range(len(sentences))]
  if threads > 1:
    pool = Pool(threads, init_worker)
    all_translations = []
    try:
      all_translations = pool.map(translate, data)
      pool.close()
      pool.join()
    except KeyboardInterrupt:
      logger.critical('Caught KeyboardInterrupt, terminating workers')
      pool.terminate()
      pool.join()
    for translations, sid in zip(all_translations, range(len(all_translations))):
      for translation in translations:
        output_translation(writer, translation, sid, 
                           output_features, with_rule_tree)
  else:
    for sid, sentence in data:
      for translation in translate((sid, sentence)):
          output_translation(writer, translation, sid, 
                             output_features, with_rule_tree)
    
  if type(source) == Reader:
    source.close()
  if type(writer) == Writer:
    writer.close() 
  
  total_time = total_timer.toc()
  sen_num = len(sentences)
  logger.info('%d sentences are translated in %f seconds (%fs/sen.)'
                    % (sen_num, total_time, total_time/sen_num))
