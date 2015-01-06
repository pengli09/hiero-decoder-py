#-*- coding: utf-8 -*-
'''
Created on Apr 21, 2014

@author: lpeng
'''
import sys
import re
import io
import ConfigParser
import logging

from ioutil import Reader

logger = logging.getLogger(__name__)


class AbsentWeightError(Exception):
  
  def __init(self, msg):
    self.msg = msg
    
  def __str__(self):
    return self.msg


class Config(object):

  def __init__(self, filename):
    with Reader(filename) as reader:
      config_lines = [re.sub(ur'^([^ \[]*) ', ur'\1=', line) for line in reader]
    config_str = u''.join(config_lines)
    
    config = ConfigParser.ConfigParser()
    config.readfp(io.StringIO(config_str))
    
    # decide feature orders
    order = {'lex-sgt':0,
             'lex-tgs':1,
             'trans-sgt':2,
             'trans-tgs':3,
             'word-count':4,
             'rule-count':5,
             'glue-rule-count':6,
             'lm':7,
             }
    
    # set other feature order
    #if config.has_option('switch', 'use-nn-feature'):
    #  self.use_neural_feature = config.getboolean('switch', 'use-nn-feature')
    #else:
    #  self.use_neural_feature = False
    #if self.use_neural_feature:
    #  order['nn-feature'] = len(order)
    #  if not config.has_section(self.NN_SECTION):
    #    raise NoSectionError('section "%s" is absent' % self.NN_SECTION)
     
    order['oov'] = len(order) # it should always be the last feature
    
    # load weights
    all_feature_names = set(order.keys())
    weights = [0]*len(order)
    for key, value in config.items('weights'):
      # DO NOT load useless weights 
      #if key == 'nn-feature' and not self.use_neural_feature:
      #  continue
      if key == 'base': # skip [DEFAULT]
        continue
      weights[order[key]] = float(value)
      all_feature_names.remove(key)
    
    weights[order['oov']] = -100
    all_feature_names.remove('oov')
    if len(all_feature_names) != 0:
      msg = 'weight(s) absent for feature(s): %s' % all_feature_names
      logger.error(msg)
      raise AbsentWeightError(msg)
    
    self.order = order
    self.weights = weights
    logger.info('weights : %s' % self.weights)
    
    # load common parameters
    self.x_beta = config.getfloat('param', 'X-beta')
    self.x_beamsize = config.getint('param', 'X-beamsize')
    
    self.s_beta = config.getfloat('param', 'S-beta')
    self.s_beamsize = config.getint('param', 'S-beamsize')
    
    self.rule_beamsize = config.getint('param', 'rule-beamsize')
    
    self.max_X_len = config.getint('param', 'max-X-len')
    
    self.epsilon = config.getint('param', 'epsilon')
    
    # load rule table
    self.rule_table_file = config.get('data', 'rules')
    
    # load language model data
    self.lm_file = config.get('data', 'lm-file')
    self.lm_order = config.getint('data', 'lm-order')
    
    self.enable_type3_glue_rule = False
    
    self.raw_config = config
    
  def get_feature_num(self):
    return len(self.order)
  
  def write(self, writer=sys.stdout):
    self.raw_config.write(writer)
    

class NoSectionError(Exception):
  
  def __init(self, msg):
    self.msg = msg
    
  def __str__(self):
    return self.msg


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
    exit(-1)
  config = Config(sys.argv[1])
  print config.weights
  print config.rule_table_file
  print config.lm_file
  print config.lm_order