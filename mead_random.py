# Standard imports
import random

def point_within_triangle(pt1, pt2, pt3):
   '''
   Random point within the triangle with vertices pt1, pt2 and pt3.
   https://stackoverflow.com/questions/47410054
   '''
   s, t = sorted([random.random(), random.random()])
   x = s*pt1[0]+(t-s)*pt2[0]+(1-t)*pt3[0]
   y = s*pt1[1]+(t-s)*pt2[1]+(1-t)*pt3[1]
   return (x, y)