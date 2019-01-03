# PyGame template.
import os 


import sys
import numpy as np

import pygame
import time
from pygame.locals import *


# lots of ugly repetitive lazy code here - sorry ! Use asdw and arrow keys - this is more of a demonstation
#of the physics, magnus effect can easily be added using cross product of angvel and velocity . 
class participant:
    def __init__(self,pos=None, vel=None, radius=5.0, mass=None):
        self.pos = (pos)
        self.vel =  (vel)
        self.r = (radius)#radius
        self.m=mass # mass 
        self.angvel=(angvel)




angvel =0.0
angmass = 1 # I should strictly be using moments of inertia but this is set arbitrarily anyway
#so this gives the same effect to some scaling factor - it simplifies things somewhat later on 
angle = 0.0
mball,mplayer=1.0,1.0 # absolute values dont really matter , it's more ratios so could remove mball
g =0.0003
grnd=800.0
rght=1440.0
lft=0.0
parties =[] 
parties.append( participant(np.array([740.0,300.0]),np.array([0.0,0.0]),20.0,1.0)) # ball 
parties.append( participant(np.array([740.0,600.0]),np.array([0.0,0.0]),40.0,8.0))# player 1
parties.append( participant(np.array([740.0,100.0]),np.array([0.0,0.0]),40.0,8.0)) # player 1
parties.append( participant()) # cross bar (-)2
parties.append( participant()) # cross bar (-)1
p = np.array([[0,0,0],[0,0,0]])




def update(dt):

  dt_rem = dt 
  handle_user_input()

  #basically each update cycle takes some time - but in that time many collisions may occur
  # so if we have 16 milliseconds allotred - we may jump forward to the first collision at 7ms,
  #then update another 2 ms to the next collision ,and so on  until all the time left is used without 
  # a collision - this accomodates some degree in variability in frame rates should it arise. 
  while dt_rem>0 :

    gamedt = dt_rem

    for i in range(3):
      result = processes[i]()
      if result[0]<gamedt:
        gamedt = result[0]
        process_party = [processes[i], result[1]]
    

    move(gamedt *0.999)

    if (gamedt<dt_rem):
    
      process_party[0](process_party[1])


    dt_rem= dt_rem -gamedt


  





  for event in pygame.event.get():
     
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit() 



def move(dt):
    global angvel 
    global angle 
    
    angle += angvel *dt 
    for i in range(3):
          parties[i].pos += parties[i].vel *dt
          parties[i].vel[1] += g*dt
       


def wall_colls(party =None):
  cf = 0.35 # coefficient friction 
  global angvel 
  e= 0.9
  if party!=None :

    if party ==parties[0] :

      maxi = (1+e)*party.vel[0]*cf

      change = (angvel*party.r + party.vel[1])/ (1+1/angmass)

      change = np.sign(change)* min (np.absolute(change),np.absolute(maxi))
      party.vel[1] -= change

      angvel -= change/(party.r* angmass )

    party.vel[0] = party.vel[0]*-e 
    return

  newdt=500
  p = parties[0]

  for i in range (3) :
    if (parties[i].vel[0]!= 0):
      dt1 = max(0,(rght - parties[i].r -parties[i].pos[0])/parties[i].vel[0])
      dt2 = max(0,(lft + parties[i].r -parties[i].pos[0])/(1*parties[i].vel[0]))
      if (dt1 ==0):
        dt1=500
      if (dt2 ==0):
        dt2=500

      dt3=  min (dt1,dt2)
      


      if dt3 < newdt :
        newdt = dt3
        p = parties[i]

  return [newdt,p] 



def floor_colls(party = None ):
  cf = 0.35 # coefficient friction 
  global angvel 
  e=0.75
  if party != None  :

    if party != parties[0]:
      e=0.05


    if (party.vel[1] < 0.05): # this is to stop floating point errors - I handle everything
       # 'dynamically' so don't artifically set v=0  so with no floating point truncations - the ball would never come to a complete 
       #halt but rather its vertical  speed would exponentially decay . 
      # setting e =1 doesn't really have a noticeable effect at low speeds and the overall 
      #total impulse exerted by the ground(when the ball is basically stationary vertically) over a set time would still be the (weight * time) to within the precision of our 
      #set threshold momentum - so no significanteffects on friction terms arise 
     
      e=1.0
    
      
   

    






    if party ==parties[0] :

      maxi = (1+e)*party.vel[1]*cf

      change = (angvel*party.r - party.vel[0])/ (1+1/angmass)

      change = np.sign(change)* min (np.absolute(change),np.absolute(maxi))
      party.vel[0] += change

      angvel -= change/(party.r* angmass ) 
    party.vel[1] = party.vel[1]*-e 
    

    return 


  p = parties[0]
  dt1,newdt=500.0,500.0

  for i in range (3) :

    if parties[i].vel[1] !=0:

      dt1 = max((grnd - parties[i].r -parties[i].pos[1])/parties[i].vel[1],0)
    
    if dt1 == 0 :
      dt1 = 500
    if dt1 < newdt :
      newdt =dt1
      p = parties[i]

  
  return [newdt,p]

def ball_colls(party=None):
  cf = 0.8 # coefficient friction  set higher for ball player collisions 
  ball= parties[0]

  def time_calc (rel_pos,rel_vel,radsum) : 
    newdt=500.0
    r=radsum
    
    x,y = rel_pos[0],rel_pos[1]
    u,v = rel_vel[0],rel_vel[1]
    determinant = (u**2+v**2)*(r**2)-(v*x-u*y)**2
    if determinant < 0 :
      return 500
    denom = u**2+v**2

    if denom!= 0:

      newdt = -(determinant**(.5)+u*x+v*y)/denom

    return newdt


  def resolve_coll (p1,u1,m1,p2,u2,m2,rsum):
    global angvel 
    e=0.95
    p21_norm = (p2-p1)/rsum

    

    u_c1 = np.dot(u1-u2,p21_norm) *p21_norm  
    v_c1 = (m1 -m2*e)/(m1+m2) * u_c1
    del_veln1 = ( v_c1-u_c1  )

    norm2del = np.dot(del_veln1,del_veln1)
    if (norm2del ) <0.00001:
      v_c1 = (m1 -m2*1.0)/(m1+m2) * u_c1
      del_veln1 = ( v_c1-u_c1  )


    del_veln2 = -del_veln1 *m1/m2

    p21_tang = np.array([p21_norm[1],-p21_norm[0]])
    u_t1 = np.dot(u1-u2,p21_tang)  
    maxi = (norm2del**0.5)*cf

    change = (angvel*parties[0].r - u_t1)/ (1+1/angmass + m1/m2)

    change = np.sign(change)* min (np.absolute(change),np.absolute(maxi))
    del_velt1= change * p21_tang
    del_velt2 = -del_velt1 *m1/m2
    angvel -= change/(parties[0].r* angmass ) 

    

    return u1 + del_veln1 + del_velt1,u2 +del_veln2+del_velt2

  if party!= None  :
  
    ball.vel , party.vel = resolve_coll(ball.pos,ball.vel,ball.m,party.pos,party.vel,party.m,party.r+ball.r)
   
    return 

  p = parties[0]
  newdt=500

  for i in range(1,3):
    dt1 = max(time_calc(parties[i].pos - ball.pos,parties[i].vel - ball.vel ,parties[i].r + ball.r ),0)
    if dt1 == 0 :
      dt1 = 500
    if dt1 < newdt :
      newdt =dt1
      p = parties[i]

  
  return [newdt,p]


def handle_user_input():
  keys=pygame.key.get_pressed()
  if keys[K_LEFT]:

    parties[1].vel[0]=parties[1].vel[0]= -0.8 #max (parties[1].vel[0]-0.1, -.8)
    
  
  if keys[K_d]:
    parties[1].vel[0]= 0.4 #min (parties[1].vel[0]+0.1, .8)

  elif keys[K_a]:
    parties[1].vel[0]=-0.4
  else :
    parties[1].vel[0]=0

  if keys[K_w]:
    parties[1].vel[1]=-0.4

  if keys[K_RIGHT]:
    parties[2].vel[0]= 0.4 #min (parties[1].vel[0]+0.1, .8)

  elif keys[K_LEFT]:
    parties[2].vel[0]=-0.4
  else :
    parties[2].vel[0]=0

  if keys[K_UP]:
    parties[2].vel[1]=-0.4


  
   




processes = [wall_colls,floor_colls,ball_colls]

def draw(screen):


  screen.fill((0, 255, 1)) # Fill the screen with black.

  for i in range(3):
      pygame.draw.circle(screen, (1,1,1), (int(parties[i].pos[0]),int(parties[i].pos[1])), int(parties[i].r))
    
  pygame.draw.line(screen,  (255,255,255), ((int)(parties[0].pos[0]-20*np.cos(angle)),(int)(parties[0].pos[1]-20*np.sin(angle))),( (int)(parties[0].pos[0]+20*np.cos(angle)),(int)(parties[0].pos[1]+20*np.sin(angle))),4)

  pygame.display.flip()



 
def runPyGame():

  pygame.init()

  width, height = int(rght), int(grnd+10)
  screen = pygame.display.set_mode((width, height))
  

  
  clock =time.time()
  while True: 
    
    dt =(time.time() - clock)*1000
    clock = time.time()
    update(dt) 
    draw(screen)
    
    
   
    #print(dt)
    

    

runPyGame()