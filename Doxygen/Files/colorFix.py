import os
global lineColor


lineColor = '#E0E0E0'

insert = 'div.line2 {\n\tfont-family: monospace, fixed;\n\tfont-size: 13px;\n\tmin-height: 13px;\n\tline-height: 1.0;\n\ttext-wrap: unrestricted;\n\twhite-space: -moz-pre-wrap; /* Moz */\n\twhite-space: -pre-wrap;     /* Opera 4-6 */\n\twhite-space: -o-pre-wrap;   /* Opera 7 */\n\twhite-space: pre-wrap;      /* CSS3  */\n\tword-wrap: break-word;      /* IE 5.5+ */\n\ttext-indent: -53px;\n\tpadding-left: 53px;\n\tpadding-bottom: 0px;\n\tmargin: 0px;\n\t-webkit-transition-property: background-color, box-shadow;\n\t-webkit-transition-duration: 0.5s;\n\t-moz-transition-property: background-color, box-shadow;\n\t-moz-transition-duration: 0.5s;\n\t-ms-transition-property: background-color, box-shadow;\n\t-ms-transition-duration: 0.5s;\n\t-o-transition-property: background-color, box-shadow;\n\t-o-transition-duration: 0.5s;\n\ttransition-property: background-color, box-shadow;\n\ttransition-duration: 0.5s;\n\tbackground-color: '+lineColor+';\n\t}\n'

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

path = "Users\\Elliott\\Documents\\UROP Stuff\\Code\\RLPy\\Doxygen\\Output\\html"
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
            l = f.readlines()
            string = "".join(l)
            newString = string.replace('div.line {', insert + '\ndiv.line {')
            f.close()
            f = open(os.path.join(root, file), mode='w')
            f.write(newString)
            f.close()
            
            
print "done"
