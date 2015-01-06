#-*- coding: utf-8 -*-
'''
Created on Apr 22, 2014

@author: lpeng
'''

def binary_search(items, key):
  lo = 0
  hi = len(items)-1
  loc = 0
  while lo <= hi:
    mid = (lo+hi) // 2
    if items[mid] > key:
      hi = mid-1
      loc = mid
    elif items[mid] < key:
      lo = mid+1
      loc = mid+1
    else:
      return mid
  return -(loc+1)