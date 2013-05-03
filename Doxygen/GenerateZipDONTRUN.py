#!/usr/bin/python
try:
    import os, subprocess, shutil, traceback
    #---------------- Name Variables ----------------------#
    REPO_NAME = 'RL-Python'
    OUTPUT_NAME = 'RLPy'

    #---------------- Checkout Repo -----------------------#
    print "Checking Out Fresh Copy of Repo"
    p = subprocess.Popen('svn checkout  svn://acl.mit.edu/acl_collab/agf/' + REPO_NAME)
    p.wait()

    #---------------- Run Doxygen -------------------------#
    print "Generating Doxygen"
    os.chdir('./' + REPO_NAME + '/Doxygen')
    p = subprocess.Popen('doxygen Doxyfile')
    p.wait()

    #---------------- Fix Line Colors ---------------------#
    print "Fixing Colors"
    os.chdir('./Files')
    p = subprocess.Popen('python colorFix.py')
    p.wait()

    #---------------- Remove Unwanted Files ---------------#
    print "Removing Files"
    f = open('./itemsNotInReleaseVersion.txt', mode='r')
    remove_list = f.readlines()
    f.close()
    file_list = []
    dir_list = []
    isDirs = False
    for i in range(1,len(remove_list)):
        item = remove_list[i]
        if item[-1:] == '\n':
            item = item[:-1]
        if item == 'DIRECTORIES':
            isDirs = True
        elif item == '':
            pass
        else:
            if isDirs:
                dir_list += [item]
            else:
                file_list += [item]	
    os.chdir('.//..//..//')
    for f in file_list:
        os.remove(f)
    for dir in dir_list:
        shutil.rmtree(dir)
        
    #---------------- Create Zip --------------------------#
    print "Zipping"
    os.chdir('.//..')
    os.rename('./' +REPO_NAME,'./' +OUTPUT_NAME)
    shutil.make_archive(OUTPUT_NAME, "zip", root_dir='.', base_dir='.//' + OUTPUT_NAME)

    #---------------- Remove Checkout ----------------------#
    print "Removing Temporary Checkout"
    shutil.rmtree('./' +OUTPUT_NAME)
    raw_input("Succsess: press enter to exit")
except Exception as e:
    traceback.print_exc()
    raw_input("Error Occured: press enter to exit")

