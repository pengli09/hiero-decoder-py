#-*- coding: utf-8 -*-
'''
Common parent class for maxent and neural features

@author: lpeng
'''

from errors import UnsupportedOperationError

class Feature(object):
 
  def __init__(self, feature_index):
    '''
    Args:
      feature_index: the index of this feature
    '''
    self.feature_index = feature_index
    
  def get_log_value(self, rule_tgt_side, src_sent,  
                    src_start, src_end,
                    src_X0_start=-1, src_X0_end=-1,
                    src_X1_start=-1, src_X1_end=-1,
                    tgt_X0_words=None, tgt_X1_words=None):
    '''
    Get the log feature value
    
    Args:
      rule_tgt_side: the target side the rule used
      src_sent: the source sentence to be translated
      src_start, src_end: the source span [src_start, src_end) is being 
        translated
      src_X0_start, src_X0_end: span [src_X0_start, src_X0_end) is 
        corresponding to source X0
      src_X1_start, src_X1_end: span [src_X1_start, src_X1_end) is 
        corresponding to source X1
      tgt_X0_words: words corresponding to target X0
      tgt_X1_words: words corresponding to target X1
      
    Return:
      log(Pr(tgt_rule_type | other information))
    '''
    raise UnsupportedOperationError('get_value() is not implemented yet')
  
  def get_feature_index(self):
    return self.feature_index
  
  def should_recombined(self, hypo0, hypo1):
    '''Check whether hypo0 and hypo1 should be recombined'''
    raise UnsupportedOperationError('should_recombined() is not implemented yet')