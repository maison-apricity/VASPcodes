#!/usr/bin/python3.6

#from Numeric import *
#from LinearAlgebra import *
from numpy import *

import sys

def read_file(name):
  old=open(name,'r')
  xvect=[]
  yvect=[]
  for line in old.readlines():
    prefield=line.split()
    xvect.append(float(prefield[0]))
    yvect.append(float(prefield[1]))
  old.close()
  return array(xvect),array(yvect)

def spline_value(a,b,c,d,x,xi):
  yi=a+b*(xi-x)+c*(xi-x)**2+d*(xi-x)**3
  return yi

#c name of file in 2-column format
name = sys.argv[1]

#c number of grid points
npt = int(sys.argv[2])

x,y=read_file(name)
nx=len(x)
ns=nx-1

a_arr=y.copy()
b_arr=zeros(ns,float)
d_arr=zeros(ns,float)
h_arr=zeros(ns,float)

for i in range(ns):
  h_arr[i] = x[i+1] - x[i]

alpha_arr=zeros(nx,float)

for i in range(1,ns):
  alpha_arr[i] = 3 * ( a_arr[i+1] - a_arr[i] ) / h_arr[i] - 3 * ( a_arr[i] - a_arr[i-1] ) / h_arr[i-1]

c_arr=zeros(nx,float)
l_arr=zeros(nx,float)
mu_arr=zeros(nx,float)
z_arr=zeros(nx,float)

l_arr[0]=1.

for i in range(1,ns):
  l_arr[i] = 2 * ( x[i+1] - x[i-1] ) - h_arr[i-1] * mu_arr[i-1]
  mu_arr[i] = h_arr[i] / l_arr[i]
  z_arr[i] = ( alpha_arr[i] - h_arr[i-1] * z_arr[i-1] ) / l_arr[i]

l_arr[-1]=1.


for j in range(nx-2,-1,-1):
  c_arr[j] = z_arr[j] - mu_arr[j] * c_arr[j+1]
  b_arr[j]=(a_arr[j+1]-a_arr[j])/h_arr[j] - h_arr[j]*(c_arr[j+1]+2*c_arr[j])/3.
  d_arr[j]=(c_arr[j+1]-c_arr[j])/(3*h_arr[j])


xx=zeros(npt,float)
if x[-1]<x[0]:
  for i in range(npt):
    xx[i]=x[0]+(x[-1]-x[0])/(npt-1)*i
    for j in range(nx-1):
      if xx[i]<=x[j]:
        indx=j
    yy=spline_value(a_arr[indx],b_arr[indx],c_arr[indx],d_arr[indx],x[indx],xx[i])
    print (xx[i],yy)
else:
  for i in range(npt):
    xx[i]=x[0]+(x[-1]-x[0])/(npt-1)*i
    for j in range(nx-1):
      if xx[i]>=x[j]:
        indx=j
    yy=spline_value(a_arr[indx],b_arr[indx],c_arr[indx],d_arr[indx],x[indx],xx[i])
    print (xx[i],yy)


  
