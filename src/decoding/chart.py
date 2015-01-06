#-*- coding: utf-8 -*-
'''
Created on Apr 21, 2014

@author: lpeng
'''
from alg import binary_search
from errors import UnexpectedStateError

class Cell(object):
  '''
  A cell of a search cube
  '''
  
  # the following functions are used for debugging
  hypo_state_checker = None # a function for checking the state of hypothesis
  cell_state_checker = None # a function for checking the state of a cell
  
  # action id and the corresponding string name
  actions = {0:'added', 1:'recombined', 2:'discarded'}
  
  def __init__(self, max_size, beta, recombination_checker):
    '''
    Args:
      max_size: max number of hypothesis stored in this cell
      beta: the hypotheses whose scores are lower than the score of the best
        hypothesis in this cell minus beta will be discarded
      recombination_checker: a function for checking whether two hypothesis
        should be recombined
    '''
    
    self.__items = []
    self.__max_size = max_size
    self.__beta = beta
    self.__recombination_checker = recombination_checker 
  
  def set_max_size(self, max_size):
    self.__max_size = max_size
    
  def get_max_size(self):
    return self.__max_size
  
  def __getitem__(self, index):
    return self.__items[index]
  
  def __len__(self):
    return len(self.__items)
  
  def __str__(self, prefix='', suffix='\n'):
    item_strs = [prefix + str(item) for item in self.__items]
    return suffix.join(item_strs)
  
  def put(self, hypothesis):
    '''
    Try to put hypothesis into this cell.
     
    Args:
      hypothesis: a hypothesis that will be put into this cell
      
    Return:
      action: 0 - the hypothesis is added to this cell, 1 - the hypothesis is
        recombined with some existing hypothesis, 2 - the hypothesis is 
        discarded
    '''
    
    action = self.__put(hypothesis)
    if self.hypo_state_checker != None:
      for hypo in self.__items:
        self.hypo_state_checker(hypo)
    if self.cell_state_checker != None:
      self.cell_state_checker(self)
    
    return action
  
  def __put(self, hypothesis):
    '''
    Try to put hypothesis into this cell.
     
    Args:
      hypothesis: a hypothesis that will be put into this cell
      
    Return:
      action: 0 - the hypothesis is added to this cell, 1 - the hypothesis is
        recombined with some existing hypothesis, 2 - the hypothesis is 
        discarded
    '''
    if len(self.__items) == 0:
      self.__items.append(hypothesis)
      return 0
    if hypothesis.score < self.__items[0].score - self.__beta:
      return 2
       
    # recombination
    for i in range(len(self.__items)):
      if self.__recombination_checker(self.__items[i], hypothesis):
        if self.__items[i].score < hypothesis.score:
          loser = self.__items[i]
          hypothesis.losers = loser.losers
          loser.losers = []
          hypothesis.losers.append(loser)
          self.__items[i] = hypothesis
          # find a new position
          for cur in range(i, 0, -1):
            if self.__items[cur].score > self.__items[cur-1].score:
              self.__items[cur], self.__items[cur-1] = (self.__items[cur-1], 
                                                        self.__items[cur])
            else:
              break
        else:
          self.__items[i].losers.append(hypothesis)
        return 1
       
    pos = binary_search(self.__items, hypothesis)
    if pos < 0:
      pos = -pos-1
      self.__items.insert(pos, hypothesis)
      if len(self.__items) > self.__max_size:
        self.__items = self.__items[0:self.__max_size]
      return 0
    else:
      msg = 'Duplicate hypothesis found:\n\t%s\n\t%s' % (self.__items[pos],
                                                         hypothesis)
      raise UnexpectedStateError(msg)
    

class Chart(object):
  '''
  A chart used in beam search
  '''
  
  def __init__(self, sentence_len, s_b, s_beta, x_b, x_beta, max_X_len, 
               recombination_checker):
    '''
    Args:
      sentence_len: the length of the sentence
      s_b: beam size of the S cell
      s_beta: beta value of S cell
      x_b: beam size of the X cell
      x_beta: beta value of X cell
      max_X_len: the max number of words an X can cover
      recombination_checker: a function for checking whether two hypotheses
        should be recombined
    '''
    # TODO: the same recombination checker should be used for 
    #       both X cell and S cell?
    self.scells = [Cell(s_b, s_beta, recombination_checker) 
                   for _ in range(sentence_len)]
    xcells = []
    for width in range(1, max_X_len+1):
      xcells.append([Cell(x_b, x_beta, recombination_checker)
                     for _ in range(sentence_len-width+1)])
    self.xcells = xcells
        
  def get_X_cell(self, width, start):
    return self.xcells[width-1][start]
  
  def get_S_cell(self, width):
    return self.scells[width-1]
  
  def add_X_item(self, width, start, hypothesis):
    return self.xcells[width-1][start].put(hypothesis)