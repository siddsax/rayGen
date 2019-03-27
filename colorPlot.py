import numpy as np
#from pylab import*
#import matplotlib as m
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib import gridspec
#from mayavi import mlab
from matplotlib import pyplot as mp
from PIL import Image
import sys
kappa = 0.0001
d = 0.1

A=20
font = {'family' : 'serif', 'weight' : 'normal', 'size' : A}
plt.rc('font', **font)
B = 19
print "CHECK"

rayleighs = ['1e6', '2e6', '5e6', '1e7', '2e7', '5e7', '1e8']

if len(sys.argv) < 2:
    print("image not given to colorize")
    exit()

im = Image.open(sys.argv[1])
arr = np.array(im)
grey = arr[:,:,0]
grey = grey / 255.0

def plot_eps_2D(TAll):

  x = np.load('x.npy')*10
  for c in range(7):
    for r in range(7):
      T = TAll[c*256+(c+1)*2 :(c+1)*256+(c+1)*2 , r*256+(r+1)*2 :(r+1)*256+(r+1)*2]

      #print "Max Value:", np.amax(np.reshape(T,256*256))
      fig, axes = plt.subplots(figsize=(10,10))
      levels = []
      print "CHECK"
      init = 9.9e8
      
      density = axes.pcolor(x,x, T, cmap='jet')#, norm = colors.LogNorm())
      
      plt.title("Rayleigh = " + rayleighs[r])
      
      axes.set_aspect(1)
      axes.set_xticks([0, 0.5, 1.0])
      axes.set_yticks([0, 0.5, 1.0])
      axes.set_xlabel('x/d')
      axes.set_ylabel('z/d')
      axes.tick_params(axis='x', which='major', pad=10)
      cb1 = fig.colorbar(density, fraction=0.05, ax=axes)#, ticks=[1e-2, 1e0, 1e2]) ###### TICKS FOR THE COLORBARS ARE DEFINED HERE
      cb1.ax.tick_params(labelsize=A)
      fig.tight_layout()
      # plt.show()
      mp.savefig('gen/foo_{0}_{1}.png'.format(r, c), bbox_inches='tight')
      plt.close('all')
  
plot_eps_2D(grey)

images = []
for i in range(7):
  im = []
  for j in range(7):
    a = np.array(Image.open("gen/foo_{0}_{1}.png".format(i, j)))
    im.append(a)
  
  im = np.concatenate(im, axis=1)
  images.append(im)

images = np.concatenate(images, axis=0)
result = Image.fromarray(images.astype(np.uint8))
result.save('out.png')

