#-*- coding: utf-8 -*-
'''
Created on Apr 21, 2014

@author: lpeng
'''
import sys
import heapq
import logging

from decoding.chart import Chart
from errors import UnsupportedOperationError, UnexpectedStateError
from rules import RuleTable
from chart import Cell

__all__ = ['CKYDecoder']

logger = logging.getLogger(__name__)
logger_checker = logging.getLogger(__name__+'.checker')
DEBUG = 0

class CKYDecoder(object):
  
  def __init__(self, config, rule_table, lm, recombination_checker,
               extra_feature_funcs, true_kbest=False,
               checking_hypo=False, expend_loser=False):
    self.stats = {}
    self.__true_kbest = true_kbest
    
    self.__feature_num = config.get_feature_num()
    self.__rule_table = rule_table
    
    self.__lm = lm
    self.__lm_feat_index = config.order['lm']
    self.__lm_weight = config.weights[config.order['lm']]
    
    self.__oov_feat_index = config.order['oov']
    self.__rule_count_feat_index = config.order['rule-count']
    self.__glue_rule_feat_index = config.order['glue-rule-count']
    self.__word_count_feat_index = config.order['word-count']
    
    self.__weights = config.weights

    self.__x_beta = config.x_beta
    self.__x_b = config.x_beamsize
  
    self.__s_beta = config.s_beta
    self.__s_b = config.s_beamsize
    
    self.__rule_b = config.rule_beamsize
    
    self.__max_X_len = config.max_X_len
    
    self.__epsilon = config.epsilon 
    
    self.__expend_loser = expend_loser
    
    self.__extra_feature_funcs = extra_feature_funcs
    self.__extra_feature_weights = []
    for func in extra_feature_funcs:
      idx = func.get_feature_index()
      self.__extra_feature_weights.append(config.weights[idx])
    
    self.__recombination_checker = recombination_checker
    
    if checking_hypo:
      self.__checking_hypo = True
      self.__config = config
      Hypothesis.feature_checker = self.__check_hypo_feature
      Cell.hypo_state_checker = self.__check_loser_state
      Cell.cell_state_checker = self.__check_cell_state
    else:
      self.__checking_hypo = False
  
  def translate(self, sentence, k=1, drop_oov=False, with_rule_tree=False):
    ''' 
    Translate a sentence.
    
    Args:
      sentence: a unicode string
      k: how many translations to output
      drop_oov: drop OOV words
      with_rule_tree: output rule tree
      
    Rreturn:
      a list of Translation
    '''
    Hypothesis.hypo_id = 0
    if len(sentence) == 0:
      return [Translation('', ), [0]*self.__feature_num, 0]
    
    chart = self.__build_chart(sentence, k)
    
    if self.__true_kbest:
      return self.__generate_true_kbest(chart, k)
    else:
      hypotheses = []
      scell = chart.get_S_cell(len(sentence))
      for i in range(len(scell)):
        hypo = scell[i]
        hypotheses.append(hypo)
        if self.__expend_loser:
          for loser in hypo.losers:
            hypotheses.append(loser)
      hypotheses.sort()
      
      translations = []
      for i in range(min(k, len(hypotheses))):
        hypo = hypotheses[i]
        translation = self.__hypo_to_translation(hypo, drop_oov, with_rule_tree)
        translations.append(translation)
      return translations
  
  def __hypo_to_translation(self, hypo, drop_oov=False, with_rule_tree=False):
    translation = ' '.join(self.__get_hypo_target_words(hypo, drop_oov))
    features = hypo.features[0:-1]
    score = hypo.score
    if with_rule_tree:
      return Translation(translation, features, score, 
                         rule_tree=hypo.get_global_rule_id_tree())
    else:
      return Translation(translation, features, score)
  
  def __get_hypo_target_words(self, hypo, drop_oov):
    if hypo.rule_id == -1: # OOV word
      if drop_oov:
        return []
      else:
        return list(hypo.left_nwords)
    else:
      rule = self.__rule_table[hypo.rule_id]
      words = list(rule.tgt)
      nonterminals = sorted(rule.nonterminal_pos)
      for i in range(len(nonterminals)-1, -1, -1):
        pos = nonterminals[i]
        words[pos:pos+1] = self.__get_hypo_target_words(hypo.pre_hypos[i], 
                                                        drop_oov)
      return words
    
  def __build_chart(self, sentence, k):
    '''
    Do cube pruning
    
    Args:
      sentence: sentence to be translated
      k: the size of k-best list
      
    Return:
      a chart
    '''    
    
    Hypothesis.id = 0 # reset hypothesis id
    sen_len = len(sentence)
    chart = Chart(sen_len,
                  self.__s_b, self.__s_beta,
                  self.__x_b, self.__x_beta,
                  self.__max_X_len,
                  self.__recombination_checker)
    
    # X chart
    for width in range(1, self.__max_X_len+1):
      for start in range(0, sen_len-width+1):
        # consider span [start, start+width]
        hyper_cube = HyperCube()
        
        # init phrase translations
        src_phrase = ' '.join(sentence[start:start+width])
        rule_ids = self.__rule_table.get_rule_ids(src_phrase)
        if len(rule_ids) == 0:
          if width == 1: # OOV word
            oov_word = src_phrase
            left_nwords = [oov_word]
            right_nwords = [oov_word]
            tgt_words = [oov_word]
            features = [0]*self.__feature_num
            # oov feature
            features[self.__oov_feat_index] = 1
            # rule count feature
            features[self.__rule_count_feat_index] = 1
            # lm feature
            hlmscore = self.__lm.get_prob([oov_word])
            features[self.__lm_feat_index] = hlmscore
            # word count feature
            features[self.__word_count_feat_index] = 1
            
            # extra feature values
            src_info = {'src_start':start, 'src_end':start + width,
                        'src_sent':sentence}
            for feat_func in self.__extra_feature_funcs:
              idx = feat_func.get_feature_index()
              features[idx] = feat_func.get_log_value(tgt_words, **src_info)
                                      
            score = 0
            for i in range(len(features)):
              score += self.__weights[i] * features[i]
              
            oov_hypo = Hypothesis(src_phrase, left_nwords, right_nwords,
                                  tgt_words, features, score, hlmscore,
                                  global_rule_id=RuleTable.OOV_RULE_GLOBAL_ID,
                                  src_info=src_info)
            chart.add_X_item(width, start, oov_hypo)
            
            logger.debug('Add OOV for span [%d, %d] - "%s"' 
                         % (start, start+1, src_phrase))
        else:
          rules = [self.__rule_table[i] for i in rule_ids]
          src_info = {'src_start':start, 'src_end':start + width,
                      'src_sent':sentence}
          xcube = InitCube(src_phrase, rules, rule_ids, self.__lm.order,
                           self.__feature_num, src_info,
                           self.__extra_feature_funcs,
                           self.__extra_feature_weights)
          hyper_cube.add_cube(xcube)
          
          if logger.level <= logging.DEBUG:
            logger.debug('rules found for span [%d, %d] - : "%s"' 
                         % (start, start+width, src_phrase))
            for __rule in rules:
              logger.debug('\t%s' % __rule)
        
        # one nonterminal
        for x_start in range(start, start+width-1+1):
          for x_end in range(x_start+1, start+width+1):
            if x_end-x_start == width:
              # at least one word should be left
              logger.debug('skip span [%d, %d] with X[%d, %d]' %
                           (start, start+width, x_start, x_end))
              continue
          
            words = sentence[start:x_start]
            words.append('|0')
            words.extend(sentence[x_end:start+width])
            src_phrase = ' '.join(words)
            rule_ids = self.__rule_table.get_rule_ids(src_phrase)
            if len(rule_ids) == 0:
              fmt = 'no rules found for span [%d, %d] with X[%d, %d] - "%s"'
              logger.debug(fmt % (start, start+width, x_start, x_end, 
                                  src_phrase))
              continue
          
            rules = [self.__rule_table[i] for i in rule_ids]

            if logger.level <= logging.DEBUG:
              fmt = 'rules found for span [%d, %d] with X[%d, %d] - "%s":'
              logger.debug(fmt % (start, start+width, x_start, x_end, 
                                  src_phrase))
              for __rule in rules:
                logger.debug('\t%s' % __rule)
            
            cell1 = chart.get_X_cell(x_end-x_start, x_start)
            if len(cell1) == 0:
              fmt = 'skip span [%d, %d] with X[%d, %d] - "%s" due to empty cell'
              logger.debug(fmt % (start, start+width, x_start, x_end, 
                                  src_phrase))
              continue
            
            src_info = {'src_start':start, 'src_end':start + width,
                        'src_X0_start':x_start, 'src_X0_end':x_end,
                        'src_sent':sentence, }
            xcube = Cube(src_phrase, rules, rule_ids,
                         self.__lm, self.__lm_feat_index, self.__lm_weight,
                         src_info, self.__extra_feature_funcs,
                         self.__extra_feature_weights, cell1)
            hyper_cube.add_cube(xcube)
      
        # two nonterminals
        for x1_start in range(start, start+width-3+1):
          for x1_end in range(x1_start+1, start+width-2+1):
            # there should be at least one word between X1 and X2
            for x2_start in range(x1_end+1, start+width-1+1):
              for x2_end in range(x2_start+1, start+width+1):
                words = sentence[start:x1_start]
                words.append('|0')
                words.extend(sentence[x1_end:x2_start])
                words.append('|1')
                words.extend(sentence[x2_end:start+width])
                src_phrase = ' '.join(words)
                
                rule_ids = self.__rule_table.get_rule_ids(src_phrase)
                if len(rule_ids) == 0:
                  fmt = ('no rules found for span [%d, %d] with '
                         'X1[%d, %d] X2[%d, %d] - "%s":')
                  logger.debug(fmt % (start, start+width, 
                                      x1_start, x1_end,
                                      x2_start, x2_end,
                                      src_phrase))
                  continue
                
                rules = [self.__rule_table[i] for i in rule_ids]
                
                if logger.level <= logging.DEBUG:
                  fmt = ('rules found for span [%d, %d] with '
                         'X1[%d, %d] X2[%d, %d] - "%s":')
                  logger.debug(fmt % (start, start+width,
                                      x1_start, x1_end,
                                      x2_start, x2_end,
                                      src_phrase))
                  for __rule in rules:
                    logger.debug('\t%s' % __rule)
                
                cell1 = chart.get_X_cell(x1_end-x1_start, x1_start)
                cell2 = chart.get_X_cell(x2_end-x2_start, x2_start)
                if len(cell1) == 0 or len(cell2) == 0:
                  fmt = ('skip span [%d, %d] with X[%d, %d] - "%s"'
                         'due to empty cell: len(cell1)=%d, len(cell2)=%d')
                  logger.debug(fmt % (start, start+width, 
                                      x_start, x_end,
                                      src_phrase,
                                      len(cell1), len(cell2)))
                  continue
                
                src_info = {'src_start':start, 'src_end':start + width,
                            'src_X0_start':x1_start, 'src_X0_end':x1_end,
                            'src_X1_start':x2_start, 'src_X1_end':x2_end,
                            'src_sent':sentence}    
                xcube = Cube(src_phrase, rules, rule_ids,
                             self.__lm, self.__lm_feat_index, self.__lm_weight,
                             src_info, self.__extra_feature_funcs,
                             self.__extra_feature_weights,
                             cell1, cell2)
                hyper_cube.add_cube(xcube)
        
        # fill X cell
        logger.debug('filling cell[X, %d, %d]' % (start, start+width))
        current_cell = chart.get_X_cell(width, start)
        while True:
          hypo = hyper_cube.next()
          if hypo == None:
            break
          if (len(current_cell) >= self.__x_b and 
              current_cell[-1].score - self.__epsilon > hypo.score): 
            break
          
          action = current_cell.put(hypo)
          if action in self.stats:
            self.stats[action] += 1
          else:
            self.stats[action] = 1
          logger.debug('\t%s -> %s' % (Cell.actions[action], hypo))
        
        if logger.level <= logging.DEBUG:
          logger.debug('Xcell[%d][%d]' % (width, start))
          xcell = chart.get_X_cell(width, start)
          for i in range(len(xcell)):
            logger.debug('\t%d -> %s' % (i, xcell[i]))
    
    # S chart
    for width in range(1, sen_len+1):
      hyper_cube = HyperCube()

      # S -> X
      if width <= self.__max_X_len:
        rule_id = self.__rule_table.get_rule_ids(RuleTable.GLUE_RULE1)[0]
        rule = self.__rule_table[rule_id]
        cell1 = chart.get_X_cell(width, 0)
        if len(cell1) == 0:
          fmt = 'skip S -> X for span [%d, %d] due to empty cell'
          logger.debug(fmt % (0, width))
        else:
          src_info = {'src_start':0, 'src_end':width, 'src_sent':sentence,
                      'src_X0_start':0, 'src_X0_end':width}
          scube1 = Type1SCube(cell1, rule, rule_id, src_info,
                              self.__extra_feature_funcs,
                              self.__extra_feature_weights)
          hyper_cube.add_cube(scube1)
          
          logger.debug('S -> X[%d, %d] activated' % (0, width))
        
      # S -> S X
      for mid in range(max(1, width-self.__max_X_len), width-1+1):
        rule_ids = self.__rule_table.get_rule_ids(RuleTable.GLUE_RULE2)
        rules = [self.__rule_table[i] for i in rule_ids]
        scell = chart.get_S_cell(mid)
        xcell = chart.get_X_cell(width-mid, mid)
        
        if len(scell) == 0 or len(xcell) == 0:
          fmt = ('skip S -> S[%d, %d] X[%d, %d] due to empty cell'
                 ': len(cell1)=%d, len(cell2)=%d')
          logger.debug(fmt % (0, mid, mid, width, 
                              len(scell), len(xcell)))
          continue
        
        src_info = {'src_start':0, 'src_end':width,
                    'src_X0_start':0, 'src_X0_end':mid,
                    'src_X1_start':mid, 'src_X1_end':width,
                    'src_sent':sentence} 
        scube2 = Type2Cube('|0 |1', rules, rule_ids,
                           self.__lm, self.__lm_feat_index, self.__lm_weight,
                           src_info, self.__extra_feature_funcs,
                           self.__extra_feature_weights, scell, xcell)
        hyper_cube.add_cube(scube2)
        
        fmt = 'S -> S[%d, %d] X[%d, %d] activated'
        logger.debug(fmt % (0, mid, mid, width))
    
      # fill S cell
      logger.debug('filling cell[S, 0, %d]' % width)
      current_cell = chart.get_S_cell(width)
      if width == sen_len and not self.__true_kbest:
        max_size = max(k, self.__s_b)
        current_cell.set_max_size(max_size)
      else:
        max_size = self.__s_b
        
      while True:
        hypo = hyper_cube.next()
        if hypo == None:
          break
        if width == sen_len:
          # add <s> </s>
          self.__finish_lm(hypo)
          if self.__expend_loser:
            for loser in hypo.losers:
              self.__finish_lm(loser)
        
        if (len(current_cell) >= max_size and 
            current_cell[-1].score - self.__epsilon > hypo.score):
          break
      
        action = current_cell.put(hypo)
        if action in self.stats:
          self.stats[action] += 1
        else:
          self.stats[action] = 1
        logger.debug('\t%s -> %s' % (Cell.actions[action], hypo))
        
      if logger.level <= logging.DEBUG:
        logger.debug('Scell[%d]' % width)
        scell = chart.get_S_cell(width)
        for i in range(len(scell)):
          logger.debug('\t%d -> %s' % (i, scell[i]))

    return chart
  
  def __generate_true_kbest(self, chart, k):
    '''
    Generate kbest translations
    '''
    raise UnsupportedOperationError('True k-best generation is not supported')

  def __finish_lm(self, hypo):
    # start symbol
    words = ['<s>'] + hypo.left_nwords 
    lm_delta = self.__lm.get_prob(words, 1)
    
    # end symbol
    if len(hypo.right_nwords) < self.__lm.order-1:
      words = ['<s>'] + hypo.right_nwords + ['</s>']
    else:
      words = hypo.right_nwords + ['</s>']
    begin = len(words)-1
    lm_delta += self.__lm.get_prob(words, begin)
    
    lm_delta -= hypo.hlmscore
    
    hypo.features[self.__lm_feat_index] += lm_delta
    hypo.score += self.__lm_weight * lm_delta
    
    if self.__checking_hypo:
      self.__check_full_lm_score(hypo) 
   
  def __check_full_lm_score(self, hypo):
    if DEBUG:
      print >> sys.stderr, '__check_full_lm_score'
    threshold = 1e-4
    tgt_words = ['<s>'] + hypo.tgt_words + ['</s>']
    expected_full_lmscore = self.__lm.get_prob(tgt_words, 1)
    actual_full_lmscore = hypo.features[self.__lm_feat_index]
    if abs(expected_full_lmscore-actual_full_lmscore) > threshold:
      logger_checker.debug('Full language model score diff > %f' % threshold)
      logger_checker.debug('\texpected: %f, actual: %f' % (expected_full_lmscore,
                                                           actual_full_lmscore))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('full language model score checking failed')
      
  def __check_hypo_feature(self, hypo, src_info):
    if DEBUG:
      print >> sys.stderr, '__check_hypo_feature'
    expected_tgt_words = self.__get_hypo_target_words(hypo, False)
    actual_tgt_words = list(hypo.tgt_words)
    if expected_tgt_words != actual_tgt_words:
      logger_checker.debug('expected_tgt_words != actual_tgt_words')
      logger_checker.debug('\texpected_tgt_words: %s' % expected_tgt_words)
      logger_checker.debug('\tactual_tgt_words: %s' % actual_tgt_words)
      
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError(expected_tgt_words != actual_tgt_words)
    
    threshold = 1e-4
    tgt_words = list(hypo.tgt_words)
    # checking language model
    expected_lmscore = self.__lm.get_prob(tgt_words)
    actual_lmscore = hypo.features[self.__lm_feat_index]
    if abs(expected_lmscore-actual_lmscore) > threshold:
      logger_checker.debug('Language model score diff > %f' % threshold)
      logger_checker.debug('\texpected: %f, actual: %f' % (expected_lmscore,
                                                           actual_lmscore))
      hypo.log_debug_str(logger_checker)
      rule_ids = hypo.get_rule_ids()
      for rule_id in rule_ids:
        if rule_id == -1:
          logger_checker.debug('%d -> OOV rule' % rule_id)
        else:
          rule = self.__rule_table[rule_id]
          logger_checker.debug('%d -> %s' % (rule_id, rule))
      raise UnexpectedStateError('language model feature checking failed')

    # checking hlmscore
    bound = min(self.__lm.order-1, len(tgt_words))
    expected_hlmscore = self.__lm.get_prob(tgt_words[0:bound])
    actual_hlmscore = hypo.hlmscore
    if abs(expected_hlmscore-actual_hlmscore) > threshold:
      logger_checker.debug('hlmscore diff > %f' % threshold)
      logger_checker.debug('expected: %f, actual: %f' % (expected_hlmscore,
                                                         actual_hlmscore))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('hlmscore feature checking failed')

    # checking feature values are correct
    actual_features = hypo.features
    expected_features = [0]*self.__config.get_feature_num()
    rule_ids = hypo.get_rule_ids()
    all_glue_rule_count = 0
    type1_glue_rule_count = 0
    for rule_id in rule_ids:
      if rule_id == -1: # OOV
        expected_features[self.__oov_feat_index] += 1
      else:
        # glue rule count
        if rule_id in self.__rule_table.glue_rule_ids:
          all_glue_rule_count += 1
        # type 1 glue rule count
        if rule_id == 0:
          type1_glue_rule_count += 1
        # accumulate features
        rule = self.__rule_table[rule_id]
        for i in range(len(rule.features)):
          expected_features[i] += rule.features[i]
    # rule-count feature
    expected_features[self.__rule_count_feat_index] = len(rule_ids)-all_glue_rule_count
    # glue-rule-count feature
    expected_features[self.__glue_rule_feat_index] = all_glue_rule_count-type1_glue_rule_count
    # word-count feature
    expected_features[self.__word_count_feat_index] = len(tgt_words)
    
    # as all the child hypotheses has been checked, we assume that extra
    # feature values of the child hypotheses are correct
    for feat_func in self.__extra_feature_funcs:
      idx = feat_func.get_feature_index()

      x0_hypo = hypo.pre_hypos[0]
      if x0_hypo is not None:
        expected_features[idx] += x0_hypo.features[idx]
        tgt_X0_words = x0_hypo.tgt_words
      else:
        tgt_X0_words = None
      
      x1_hypo = hypo.pre_hypos[1]
      if x1_hypo is not None:
        expected_features[idx] += x1_hypo.features[idx]
        tgt_X1_words = x1_hypo.tgt_words
      else:
        tgt_X1_words = None
            
      if hypo.rule_id >= 0:
        rule = self.__rule_table[hypo.rule_id]
        rule_tgt_side = rule.tgt
      else:
        # OOV
        rule_tgt_side = hypo.tgt_words
      
      value = feat_func.get_log_value(rule_tgt_side, tgt_X0_words=tgt_X0_words,
                                      tgt_X1_words=tgt_X1_words, **src_info)
      expected_features[idx] += value
  
    # checking features
    for i in range(len(expected_features)):
      # skip language model score score
      if i == self.__lm_feat_index:
        continue
      
      diff = abs(expected_features[i]-actual_features[i])
      if diff > threshold:
        logger_checker.debug('diff of feature[%d] > %f' % (i, threshold))
        logger_checker.debug('\texpected: %f, actual: %f' % (expected_features[i],
                                                             actual_features[i]))
        hypo.log_debug_str(logger_checker)
        raise UnexpectedStateError('feature value checking failed')
    
    # checking feature num
    if len(hypo.features) != len(self.__config.weights):
      len1 = len(hypo.features)
      len2 = len(self.__config.weights)
      logger_checker.debug('len(hypo.features) != len(self.__config.weights)')
      logger_checker.debug('\t%d != %d' % (len1, len2))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('feature number checking failed')
      
    # checking total score
    expected_score = 0
    for i in range(len(hypo.features)):
      expected_score += hypo.features[i]*self.__config.weights[i]
    actual_score = hypo.score
    if abs(expected_score-actual_score) > threshold:
      logger_checker.debug('score diff > %f' % threshold)
      logger_checker.debug('\texpected: %f, actual: %f' % (expected_score,
                                                           actual_score))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('total score checking failed')
    
    # checking left_nwords and right_nwords
    word_num = min(self.__lm.order-1, len(tgt_words))
    left_nwords = tgt_words[0:word_num]
    if left_nwords != hypo.left_nwords:
      logger_checker.debug('left_nwords is not correct')
      logger_checker.debug('\texpected: %s, actual: %s' % (left_nwords,
                                                           hypo.left_nwords))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('left_nwords checking failed')
    right_nwords = tgt_words[-word_num:]
    if right_nwords != hypo.right_nwords:
      logger_checker.debug('right_nwords is not correct')
      logger_checker.debug('\texpected: %s, actual: %s' % (right_nwords,
                                                           hypo.right_nwords))
      hypo.log_debug_str(logger_checker)
      raise UnexpectedStateError('right_nwords checking failed')
      
  def __check_loser_state(self, hypo):
    if DEBUG:
      print >> sys.stderr, '__check_loser_state'
    for loser in hypo.losers:
      if hypo.score < loser.score:
        logger_checker.debug('score of loser is higher than that of hypo')
        logger_checker.debug('\tloser: %s' % loser)
        logger_checker.debug('\thypo : %s' % hypo)
        raise UnexpectedStateError('score of loser is higher than that of hypo')
      if hypo.left_nwords != loser.left_nwords:
        logger_checker.debug('loser.left_nwords != hypo.left_nwords')
        logger_checker.debug('\tloser: %s' % loser)
        logger_checker.debug('\thypo : %s' % hypo)
        raise UnexpectedStateError('loser.left_nwords != hypo.left_nwords')
      if hypo.right_nwords != loser.right_nwords:
        logger_checker.debug('loser.right_nwords != hypo.right_nwords')
        logger_checker.debug('\tloser: %s' % loser)
        logger_checker.debug('\thypo : %s' % hypo)
        raise UnexpectedStateError('loser.right_nwords != hypo.right_nwords')
  
  def __check_cell_state(self, cell):
    if DEBUG:
      print >> sys.stderr, '__check_cell_state'
    for i in range(1, len(cell)):
      pre_hypo = cell[i-1]
      hypo = cell[i]
      if hypo.score > pre_hypo.score:
        logger_checker.debug('hypotheses in this cell is out of order')
        logger_checker.debug('\t%s', cell)
        raise UnexpectedStateError('hypotheses is out of order')
    

class Translation(object):
  
  def __init__(self, translation, features, score, rule_tree=None,
               trace_data=None):
    self.translation = translation
    self.features = features
    self.score = score
    if rule_tree != None:
      self.rule_tree = rule_tree
    if trace_data != None:
      self.trace_data = trace_data
    
  def __str__(self):
    features = ' '.join([str(f) for f in self.features])
    return '%s ||| %s ||| %f' % (self.translation, features, self.score)
 
  
class Hypothesis(object):
  
  hypo_id = 0
  feature_checker = None
  
  def __init__(self, src, left_nwords, right_nwords, tgt_words, features, score, 
               hlmscore, rule_id=-1, left_pre=None, right_pre=None,
               global_rule_id=None, src_info=None):
    '''
    Args:
      Note: n is the language model order
      src: source side of the rule
      left_nwords: the left most n words of this hypothesis
      right_nwords: the right most n words of this hypothesis
      tgt_words: the translation represented by this hypothesis
      features: feature values
      score: total score
      hlmscore: the langauge model score of the left n-1 words
      rule_id: the index of the rule in the rule table, -1 for OOV rule
      left_pre: X0 hypothesis
      right_pre: X1 hypothesis
      global_rule_id: the index of the rule in the global rule table
      src_info: for debugging purpose only
    '''
    self.src = src
    self.hypo_id = Hypothesis.hypo_id
    Hypothesis.hypo_id += 1
    self.left_nwords = left_nwords
    self.right_nwords = right_nwords
    self.tgt_words = tgt_words
    self.features = features
    self.score = score
    self.hlmscore = hlmscore
    self.rule_id = rule_id
    self.pre_hypos = [left_pre, right_pre]
    self.losers = [] # not sorted
    self.global_rule_id = global_rule_id
    
    if self.feature_checker != None:
      self.feature_checker(self, src_info)

  def __cmp__(self, other):
    if other == None: # for heapq.*
      return -1
    ret = -cmp(self.score, other.score)
    if ret == 0:
      return self.hypo_id - other.hypo_id
    else:
      return ret

  def get_rule_ids(self):
    rule_ids = [self.rule_id]
    for pre_hypo in self.pre_hypos:
      if pre_hypo == None:
        continue
      rule_ids.extend(pre_hypo.get_rule_ids());
    return rule_ids;
  
  def get_global_rule_id_tree(self):
    tree = []
    if self.rule_id == -1:
      node_tuple = [(self.hypo_id, self.global_rule_id, tuple(self.left_nwords))]
    else:
      node_tuple = [(self.hypo_id, self.global_rule_id)]
    for pre_hypo in self.pre_hypos:
      if pre_hypo == None:
        continue
      tree.extend(pre_hypo.get_global_rule_id_tree())
      node_tuple.append(pre_hypo.hypo_id)
    tree.append(tuple(node_tuple))
    return tree

  def __str__(self):
    if self.pre_hypos[0] == None:
      left_pre = -1
    else:
      left_pre = self.pre_hypos[0].hypo_id
      
    if self.pre_hypos[1] == None:
      right_pre = -1
    else:
      right_pre = self.pre_hypos[1].hypo_id
          
    return (('id: %d, left_nwords: "%s", right_nwords: "%s", '
             'rule_id: %s, left_pre: %d, right_pre: %d, '
             'score: %f, hlmscore: %f, features: %s')
             % (self.hypo_id, 
                ' '.join(self.left_nwords), ' '.join(self.right_nwords),
                self.rule_id, left_pre, right_pre, 
                self.score, self.hlmscore, self.features))
  
  def log_debug_str(self, logger, level=0):
    logger.debug('%s%s' % ('  '*level, self))
    for pre_hypo in self.pre_hypos:
      if pre_hypo == None:
        continue
      pre_hypo.log_debug_str(logger, level+1)


class HyperCube(object):
 
  class Item(object):
    
    def __init__(self, hypo, cube_index):
      self.hypo = hypo
      self.cube_index = cube_index
      
    def __cmp__(self, other):
      if other == None: # for heapq.*
        return -1
      ret = cmp(self.hypo, other.hypo)
      if ret == 0:
        return self.cube_index - other.cube_index
      else:
        return ret

  def __init__(self):
    self.__items = []
    self.__cubes = []
    
  def add_cube(self, cube):
    new_hypo = cube.next()
    if new_hypo == None:
      return
    heapq.heappush(self.__items, self.Item(new_hypo, len(self.__cubes)))
    self.__cubes.append(cube)
    
  def next(self):
    if len(self.__items) == 0:
      return None
    else:
      item = heapq.heappop(self.__items)
      new_hypo = self.__cubes[item.cube_index].next()
      if new_hypo != None:
        heapq.heappush(self.__items, self.Item(new_hypo, item.cube_index))
      return item.hypo
   

  class FakeRuleTreeNode(object):
    
    def __init__(self, vrule):
      self.rule = vrule
    
    def is_leaf(self):
      return len(self.rule.src_nonterminal_pos) == 0


class InitCube(object):
  
  def __init__(self, src, rules, rule_ids, lm_order, feature_num,
               src_info, extra_feature_funcs, extra_feature_weights):
    '''
    rules should be sorted
    '''
    self.__src = src
    self.__rules = rules
    self.__rule_ids = rule_ids
    self.__lm_order = lm_order
    self.__cur = 0
    self.__feature_num = feature_num
    self.__src_info = src_info
    self.__feat_funcs = extra_feature_funcs
    self.__feat_weights = extra_feature_weights
    
  def next(self):
    if self.__cur >= len(self.__rules):
      return None
    else:
      index = self.__cur
      self.__cur += 1
      return self.__rule_to_hypo(self.__rules[index], self.__rule_ids[index])

  def __rule_to_hypo(self, rule, rule_id, vrule=None):
    nwords = min(self.__lm_order-1, len(rule.tgt))
    left_nwords = rule.tgt[0:nwords]
    right_nwords = rule.tgt[-nwords:]
    tgt_words = list(rule.tgt)
    features = [0]*self.__feature_num
    features[0:len(rule.features)] = rule.features
    score = rule.score
    hlmscore = rule.hlmscore
    
    # extra feature values
    for feat_func, weight in zip(self.__feat_funcs, self.__feat_weights):
      idx = feat_func.get_feature_index()
      features[idx] = feat_func.get_log_value(tgt_words, **self.__src_info)
      score += features[idx] * weight
          
    return Hypothesis(self.__src, left_nwords, right_nwords, tgt_words,
                      features, score, hlmscore, rule_id,
                      global_rule_id=rule.global_rule_id,
                      src_info=self.__src_info)


class Cube(object):
  
  class Item(object):
    
    def __init__(self, hypo, signature):
      self.hypo = hypo
      self.signature = signature
      
    def __cmp__(self, other):
      if other == None: # for heapq.*
        return -1
      ret = cmp(self.hypo, other.hypo)
      if ret == 0:
        return cmp(self.signature, other.signature)
      else:
        return ret
  
  def __init__(self, src, rules, rule_ids, lm, lm_feat_index, lm_weight,
               src_info, extra_feature_funcs, extra_feature_weights,
               cell1, cell2=None):
    self.__visited = set()
    self.__items = []
    
    self.__src = src
    self.__rules = rules
    self.__rule_ids = rule_ids
    self.__lm = lm
    self.__lm_feat_index = lm_feat_index
    self.__lm_weight = lm_weight
    self.__src_info = src_info
    self.__feat_funcs = extra_feature_funcs
    self.__feat_weights = extra_feature_weights

    if cell2 != None: # two nonterminals
      self.__cells = [cell1, cell2]
      self.__build_and_add_hypo((0, 0, 0))  
    else:
      self.__cells = [cell1]
      self.__build_and_add_hypo((0, 0))
  
  def next(self):
    if len(self.__items) == 0:
      return None
    else:
      next_item = heapq.heappop(self.__items)
      self.__expand(next_item)
      return next_item.hypo
      
  def __expand(self, item):
    for i in range(len(item.signature)):
      new_signature = list(item.signature)
      new_signature[i] += 1
      if i == 0 and new_signature[i] >= len(self.__rules):
        continue 
      if i != 0 and new_signature[i] >= len(self.__cells[i-1]):
        continue
      
      new_signature = tuple(new_signature)
      self.__build_and_add_hypo(new_signature)
      
  def __build_and_add_hypo(self, signature):
    if signature in self.__visited:
      return
      
    rule = self.__rules[signature[0]]
    rule_id = self.__rule_ids[signature[0]]
    
    if len(signature) == 3:
      if rule.nonterminal_pos[0] < rule.nonterminal_pos[1]:
        first, second = 1, 2
      else:
        first, second = 2, 1
      hypo1 = self.__cells[first-1][signature[first]]
      hypo2 = self.__cells[second-1][signature[second]]
      
      pre_hypos = [hypo1, hypo2]
    else:
      hypo1 = self.__cells[0][signature[1]]
      hypo2 = None
      pre_hypos = [hypo1]

    tgt_words = list(rule.tgt)
    nonterminals = sorted(rule.nonterminal_pos)
    for i in range(len(nonterminals)-1, -1, -1):
      pos = nonterminals[i]
      tgt_words[pos:pos+1] = pre_hypos[i].tgt_words

    n = min(self.__lm.order-1, len(tgt_words))
    left_nwords = tgt_words[0:n]
    right_nwords = tgt_words[-n:]
      
    # update features
    features = list(hypo1.features)
    if hypo2 != None:
      for i in range(len(features)):
        features[i] += hypo2.features[i]
    for i in range(len(rule.features)):
      features[i] += rule.features[i]
    
    # update lmscore
    new_hlmscore = self.__lm.get_prob(left_nwords)
    hlmscore_delta = new_hlmscore - rule.hlmscore - hypo1.hlmscore
    if hypo2 != None:
      hlmscore_delta -= hypo2.hlmscore
    actual_lmscore_delta = self.__get_actual_lm_delta(rule, hypo1, hypo2)
    lmscore_delta = hlmscore_delta + actual_lmscore_delta   
    features[self.__lm_feat_index] += lmscore_delta
    
    # update score
    score = rule.score + hypo1.score
    if hypo2 != None:
      score += hypo2.score
    score += self.__lm_weight*lmscore_delta
    
    # update extra feature values and score
    for feat_func, feat_weight in zip(self.__feat_funcs, self.__feat_weights):
      idx = feat_func.get_feature_index()
      rule_tgt_side = rule.tgt
      tgt_X0_words = hypo1.tgt_words
      tgt_X1_words = None if hypo2 is None else hypo2.tgt_words
      value = feat_func.get_log_value(rule_tgt_side, 
                                      tgt_X0_words=tgt_X0_words,
                                      tgt_X1_words=tgt_X1_words,
                                      **self.__src_info)
      features[idx] += value
      score += value * feat_weight
    
    new_hypo = Hypothesis(self.__src, left_nwords, right_nwords,
                          tgt_words, features, score,
                          new_hlmscore, rule_id, hypo1, hypo2,
                          global_rule_id=rule.global_rule_id,
                          src_info=self.__src_info)
    new_item = self.Item(new_hypo, signature)

    heapq.heappush(self.__items, new_item)
    self.__visited.add(signature)
  
  def __get_actual_lm_delta(self, rule, hypo1, hypo2=None):
    if (len(rule.nonterminal_pos) == 2 
        and rule.nonterminal_pos[0] > rule.nonterminal_pos[1]):
      first, second = 1, 0 # from left to right
    else:
      first, second = 0, 1
    
    order = self.__lm.order
    lm_delta = 0
    # first nonterminal
    contexts = []
    first_X_pos = rule.nonterminal_pos[first]
    last_X_pos = first_X_pos
    # add right most n-1 words to contexts 
    context_num = min(order-1, first_X_pos)
    contexts.extend(rule.tgt[first_X_pos-context_num:first_X_pos])
    # update probabilities only if there are more than n words
    if len(contexts) + len(hypo1.left_nwords) >= order:
      # at most n-1 words are considered, and the max size of hypo1.left_nwords
      # is n-1, so it is safe to add all of hypo1.left_nwords to words
      words = contexts + hypo1.left_nwords
      lm_delta += self.__lm.get_prob(words, begin=order-1)
    # condition 1: len(hypo1.right_nwords) < n-1, hypo1.right_nwords will be
    #              the entire string corresponding to X
    # condition 2: len(hypo1.right_nwords) = n-1, the other words in contexts
    #              will be ignored automatically
    contexts.extend(hypo1.right_nwords)
    
    # middle string
    if len(rule.nonterminal_pos) == 2:
      second_X_pos = rule.nonterminal_pos[second]
      last_X_pos = second_X_pos
      # length of the middle string
      mid_str_word_num = second_X_pos-(first_X_pos+1)
      if mid_str_word_num != 0: # middle string is not empty
        # only the left most n-1 words are considered 
        v_word_num = min(order-1, mid_str_word_num)
        # update probabilities only if there are more than n words
        if mid_str_word_num + len(contexts) >= order:
          start_pos = first_X_pos+1 # start pos of middle string
          words = (contexts[-min(order-1, len(contexts)):] #right most n-1 words
                   + rule.tgt[start_pos:start_pos+v_word_num])
          # note that contexts is clipped, so begin is always order-1
          lm_delta += self.__lm.get_prob(words, begin=order-1)
        end_pos = second_X_pos
        contexts.extend(rule.tgt[end_pos-v_word_num:end_pos])
      
    # second nonterminal
    if len(rule.nonterminal_pos) == 2: 
      # update probabilities only if there are more than n words     
      if len(contexts) + len(hypo2.left_nwords) >= order:
        # at most n-1 words are considered, and the max size of 
        # hypo2.left_nwords is n-1, so it is safe to add all of
        # hypo2.left_nwords to words
        words = (contexts[-min(order-1, len(contexts)):] # right most n-1 words
                 + hypo2.left_nwords)
        # note that contexts is clipped, so begin is always order-1
        lm_delta += self.__lm.get_prob(words, begin=order-1)
      contexts.extend(hypo2.right_nwords)
      
    # last string
    start_pos = last_X_pos+1
    last_str_word_num = len(rule.tgt)-start_pos
    if len(contexts) + last_str_word_num >= order:
      v_word_num = min(order-1, last_str_word_num)
      words = (contexts[-min(order-1, len(contexts)):] # right most n-1 words
               + rule.tgt[start_pos:start_pos+v_word_num])
      # note that contexts is clipped, so begin is always order-1
      lm_delta += self.__lm.get_prob(words, begin=order-1)
    
    return lm_delta
  

class Type1SCube(object):
  '''
  S -> X
  '''

  def __init__(self, cell, rule, rule_id, src_info, extra_feature_funcs,
               extra_feature_weights):
    self.__cur = 0
    self.__cell = cell
    self.__rule = rule
    self.__rule_id = rule_id
    self.__src_info = src_info
    self.__feat_funcs = extra_feature_funcs
    self.__feat_weights = extra_feature_weights
    
  def next(self):
    if self.__cur >= len(self.__cell):
      return None
    else:
      xhypo = self.__cell[self.__cur]
      self.__cur += 1
      features = list(xhypo.features)
      for i in range(len(self.__rule.features)):
        features[i] += self.__rule.features[i]
      score = xhypo.score + self.__rule.score
      
      # update extra feature values and score
      for feat_func, feat_weight in zip(self.__feat_funcs, self.__feat_weights):
        idx = feat_func.get_feature_index()
        rule_tgt_side = ['|0']
        tgt_X0_words = xhypo.tgt_words
        value = feat_func.get_log_value(rule_tgt_side, tgt_X0_words=tgt_X0_words,
                                        **self.__src_info)
        features[idx] += value
        score += value * feat_weight
        
      return Hypothesis('|0', list(xhypo.left_nwords), list(xhypo.right_nwords),
                        list(xhypo.tgt_words), features, score, xhypo.hlmscore,
                        self.__rule_id, left_pre=xhypo, 
                        global_rule_id=self.__rule.global_rule_id,
                        src_info=self.__src_info) 


class Type2Cube(Cube):
  pass