import glob
import sys
import string
import os


# dico  de fichier
listFile  = {}
# liste modif include
listInclude = {}

pathh = ['../reflow', '../test']
for xpath in pathh:
    for path, subdirs, files in os.walk(xpath):
        for name in files:
            print " path", path , "/" , name.rstrip()
            fp =open(path+"/"+name.rstrip(),'r')   
            dlines = fp.readlines()    
            fp.close()
            fp =open(path+"/"+name.rstrip(),'w')  
            for l in dlines:
                if (l.find("Eclipse Public License")):
                    l= string.replace(l,"Eclipse Public License","GNU Lesser General Public License (GNU LGPL)")
                fp.write(l)
            fp.close()

"""                   
                if (l.find("2015") !=-1):      
                 l = string.replace(l,"2015","2016") 
                if (l.find("FiME") !=-1):
                 l = string.replace(l,"FiME","EDF")
                if (l.find("FIME") !=-1):
                 l = string.replace(l,"FIME","EDF")
"""
