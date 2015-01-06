#-*- coding: utf-8 -*-
'''
Created on Apr 21, 2014

@author: lpeng
'''
import operator
import logging
import gc

from marisa_trie import RecordTrie

from ioutil import Reader, Writer
from timeutil import Timer

logger = logging.getLogger(__name__)

DEBUG = 0
class Rule(object):
  '''
  A translation rule
  '''

  def __init__(self, src, tgt, nonterminal_pos, features, global_rule_id):
    '''
    Args:
      src: source side of the rule
      tgt: target side of the rule
      non_terminal_pos: the non_terminal_pos[i] element of tgt is Xi
      features: feature values of this rule
      global_rule_id: rule id in the global rule table
    '''
    if DEBUG:
      self.src = src
    self.tgt = tgt
    self.nonterminal_pos = nonterminal_pos
    self.features = features
    self.score = 0
    self.hlmscore = 0
    self.global_rule_id = global_rule_id
    
  def __cmp__(self, other):
    ret = -cmp(self.score, other.score)
    if ret == 0:
      return cmp(self.tgt, other.tgt)
    else:
      return ret
    
  def __str__(self):
    tgt = ' '.join(self.tgt)
    if DEBUG:
      fmt = 'src: "%s", tgt: "%s", features: %s, score: %f, hlmscore: %f'
      return fmt % (self.src, tgt, self.features, self.score, self.hlmscore)
    else:
      fmt = 'tgt: "%s", features: %s, score: %f, hlmscore: %f'
      return fmt % (tgt, self.features, self.score, self.hlmscore)
      
    
class RuleTable(object):
  '''
  A table for storing translation rules
  '''
  
  COPY_RULE_ID_RANGE = []
  # S -> X
  GLUE_RULE1 = '|0'
  GLUE_RULE1_GLOBAL_ID = -1
  # S -> S X
  GLUE_RULE2 = '|0 |1'
  GLUE_RULE2_GLOBAL_ID = -2
  # S -> <S X; X S>
  GLUE_RULE3_GLOBAL_ID = -3
  # OOV
  OOV_RULE_GLOBAL_ID = -3
  
  def __init__(self):
    self._rules = [] # rules
    self._idranges = None # the index range of rules corresponding to a source str.
    self.glue_rule_ids = None # for checking feature values
    
  def dump(self, filename, detailed=False):
    '''
    Dump rules stored in this table to a file
    
    Args:
      filename: file name
      detailed: True - output detailed information
    '''
    if detailed:
      with Writer(filename) as writer:
        ranges = sorted(self._idrange.iteritems(), key=operator.itemgetter(1))
        for key, _range in ranges:
          writer.write('key: %s, range: [%d, %d]\n' 
                        % (key, _range[0], _range[1]))
          for i in range(_range[0], _range[1]+1):
            writer.write('\t%s\n' % self._rules[i])
    else:
      with Writer(filename) as writer:
        for rule in self._rules:
          writer.write('%s\n' % rule)
  
  @classmethod
  def load(cls, filename, lm, config):
    logger.info('Loading rule table from "%s"...' % filename)
    timer = Timer()
    timer.tic()
    table = cls.__load_rules(filename, lm, config)
    logger.info('Rule table loaded in %f seconds.' % timer.toc())
    return table
  
  @classmethod
  def __load_rules(cls, filename, lm, config):
    '''
    Load rule table from filename
    
    Args:
      filename: the name of the file that stores the rules
      lm: language model
      config: an instance of Config
      
    Return:
      a RuleTable
    '''
    
    feature_num = config.get_feature_num()
    glue_rule_index = config.order['glue-rule-count']
    max_rule_num = config.rule_beamsize
    
    table = RuleTable()
    keys = []
    ranges = []
    
    idx = 0
    # glue rules
    # S -> X
    # type 1 glue rule is not counted
    features = [0]*feature_num
    glue_rule1 = Rule('|0', ['|0'], [0], features, cls.GLUE_RULE1_GLOBAL_ID)
    table._rules.append(glue_rule1)
    keys.append(cls.GLUE_RULE1.decode('utf-8'))
    ranges.append((idx, idx))
    idx += 1
    
    # S -> S X
    features = [0]*feature_num
    features[glue_rule_index] = 1
    glue_rule2 = Rule('|0 |1', ['|0', '|1'], [0, 1], features, 
                      cls.GLUE_RULE2_GLOBAL_ID)
    glue_rule2.score = config.weights[config.order['glue-rule-count']]
    table._rules.append(glue_rule2)
    idx += 1
    
    if config.enable_type3_glue_rule:
      # S -> <S X; X S>
      features = [0]*feature_num
      features[glue_rule_index] = 1
      glue_rule3 = Rule('|0 |1', ['|1', '|0'], [1, 0], features, 
                        cls.GLUE_RULE3_GLOBAL_ID)
      glue_rule3.score = config.weights[config.order['glue-rule-count']]
      table._rules.append(glue_rule3)
      idx += 1

    keys.append(cls.GLUE_RULE2.decode('utf-8'))
    ranges.append((1, idx-1))
    
    table.glue_rule_ids = tuple(i for i in range(idx))
    
    # normal rules
    with Reader(filename) as reader:
      last_src = None
      current_rules = []
      for rule_str in reader:
        parts = rule_str.strip().split(' ||| ')
        src = parts[0]
        tgt = parts[1].split(' ')
        nonterminal_pos = []
        for tword, pos in zip(tgt, range(len(tgt))):
          if tword[0] == '|':
            if len(nonterminal_pos) == 0:
              nonterminal_pos.append(pos)
            else:
              index = int(tword[1:])
              nonterminal_pos.insert(index, pos)
        features = [float(f) for f in parts[2].split(' ')]
        features.append(len(tgt)-len(nonterminal_pos)) # word number
        features.append(1) # rule count
        features.append(0) # glue rule count
        if len(parts) >= 4:
          global_rule_id = int(parts[3])
          rule = Rule(src, tgt, nonterminal_pos, features, global_rule_id)
        else:
          rule = Rule(src, tgt, nonterminal_pos, features, idx)
        lmscore, hlmscore = cls.__get_lm_scores(rule, lm) 
        features.append(lmscore) # lm score
        rule.hlmscore = hlmscore
        
        if last_src == None or src == last_src:
          current_rules.append(rule)
          last_src = src
        else:
          cls.__update_table(table, keys, ranges, last_src, current_rules, 
                             config, max_rule_num)
          current_rules = [rule]
          last_src = src
        idx += 1
        
      cls.__update_table(table, keys, ranges, last_src, current_rules, 
                         config, max_rule_num)
      
      table._idranges = RecordTrie('<II', zip(keys, ranges))
      del keys
      del ranges
      gc.collect()
    return table
  
  @classmethod
  def __update_table(cls, table, keys, ranges, src, rules, config, max_rule_num):
    if len(rules) == 0:
      return
    
    cls.__score_rules(rules, config)
    rules.sort()
    if len(rules) > max_rule_num:
      rules = rules[0:max_rule_num]
    start = len(table._rules)
    table._rules.extend(rules)
    end = len(table._rules)
    
    keys.append(src.decode('utf-8'))
    ranges.append((start, end-1))
    
  @classmethod
  def __score_rules(cls, rules, config):
    for rule in rules:
      score = 0
      for i in range(len(rule.features)):
        score += rule.features[i]*config.weights[i]
      rule.score = score
  
  @classmethod
  def __get_lm_scores(cls, rule, lm):
    actual_lmscore = 0
    hlmscore = 0
    nonterminals = sorted(rule.nonterminal_pos)
    ranges = []
    start = 0
    for end in nonterminals:
      if start < end:
        ranges.append((start, end))
      start = end+1
    if start < len(rule.tgt):
      ranges.append((start, len(rule.tgt)))
    
    for start, end in ranges:
      bound = min(lm.order-1, end-start)
      substr = rule.tgt[start:end]
      hlmscore += lm.get_prob(substr, 0, bound)
      actual_lmscore += lm.get_prob(substr, bound)
    lmscore = actual_lmscore + hlmscore    
    return lmscore, hlmscore
    
  def get_rule_ids(self, source):
    source = source.decode('utf-8')
    ids = self._idranges.get(source, [self.COPY_RULE_ID_RANGE])[0]
    if len(ids) == 0:
      return ids
    else:
      return range(ids[0], ids[1]+1) 
  
  def __getitem__(self, index):
    return self._rules[index]