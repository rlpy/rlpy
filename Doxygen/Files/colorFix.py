import os
global lineColor


lineColor = '#C0C0C0'

def fixLineColors(f):
    write = False
    l = f.readlines()
    n = 0
    for i in range(len(l)):
        string = l[i]
        if '<div class="line">' in string:
            if (n%2 == 1):
                 write = True
                 x = string.replace('<div class="line">', '<div class="line2">')
                 l[i] = x
            n += 1
    s = "".join(l)
    return s, write

path = "Users\\Elliott\\Documents\\UROP Stuff\\Code\\RL-Python\\Doxygen\\Output\\html"
directory = os.path.join("c:\\",path)
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".html"):
           
           f = open(os.path.join(root, file), mode='r+')
           #f = open("c:\\"+path+"\\"+file, mode='r+')
           s, write = fixLineColors(f)
           if write:
               f.close()
               f = open(os.path.join(root, file), mode='w')
               f.write(s)
           f.close()
        if file.endswith("doxygen.css"):
            f = open(os.path.join(root, file), mode='r+')
            string = 'div.line2 {\nfont-family: monospace, fixed;\nfont-size: 13px;\nmin-height: 13px;\nline-height: 1.0;\ntext-wrap: unrestricted;\nwhite-space: -moz-pre-wrap; /* Moz */\nwhite-space: -pre-wrap;     /* Opera 4-6 */\nwhite-space: -o-pre-wrap;   /* Opera 7 */\nwhite-space: pre-wrap;      /* CSS3  */\nword-wrap: break-word;      /* IE 5.5+ */\ntext-indent: -53px;\npadding-left: 53px;\npadding-bottom: 0px;\nmargin: 0px;\n-webkit-transition-property: background-color, box-shadow;\n-webkit-transition-duration: 0.5s;\n-moz-transition-property: background-color, box-shadow;\n-moz-transition-duration: 0.5s;\n-ms-transition-property: background-color, box-shadow;\n-ms-transition-duration: 0.5s;\n-o-transition-property: background-color, box-shadow;\n-o-transition-duration: 0.5s;\ntransition-property: background-color, box-shadow;\ntransition-duration: 0.5s;\nbackground-color: '+lineColor+';\n}\n'
            f.write(string)
            f.close()
            
            
print "done"
