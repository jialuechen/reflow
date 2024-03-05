import sys
import os
import fileinput
import re

stringToReplace  = sys.argv[1]
print " stringToReplace " , stringToReplace
newString  = sys.argv[2]
print " newString " , newString
extention =sys.argv[3]
print " extention " , extention

pathh = ['../libflow', '../test']
for xpath in pathh:
    for path, subdirs, files   in os.walk(xpath):
        for name in files:
            if name.endswith("."+extention):
                print " file " , name
                fp =open(path+"/"+name.rstrip(),'r')   
                dlines = fp.readlines()    
                fp.close()
                fp =open(path+"/"+name.rstrip(),'w')  
                for l in dlines:  
                    l= re.sub(stringToReplace,newString,l)
                    fp.write(l)
                fp.close()


 
