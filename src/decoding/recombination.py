#-*- coding: utf-8 -*-
'''

@author: lpeng
'''

class CoreFeatureRecombinationChecker(object):
  
  def __call__(self, hypo0, hypo1):
    '''
    hypo0 and hypo1 should cover the same span
    return True if hypo0 and hypo1 should be recombined
    '''  
    if hypo0.left_nwords != hypo1.left_nwords:
      return False
    if hypo0.right_nwords != hypo1.right_nwords:
      return False
    return True
  

class CombinedRecombinationChecker(object):
  '''Wrap multiple recombination checkers.
     CoreFeatureRecombinationChecker is always included.
  '''
  
  def __init__(self, extra_feature_funcs):
    '''
    Args:
      extra_feature_funcs: a list of extra feature functions beside 
        core features
    '''
    self.__checkers = []
    for func in extra_feature_funcs:
      checker = lambda hypo0, hypo1: func.should_recombined(hypo0, hypo1)
      self.__checkers.append(checker)
    self.__checkers.append(CoreFeatureRecombinationChecker())
  
  def __call__(self, hypo0, hypo1):
    for checker in self.__checkers:
      if not checker(hypo0, hypo1):
        return False
    return True