#!/usr/bin/python

# Developed by D. Elliott Williams on May 3rd, 2013 at MIT

# Checks out a temporary clean copy of the RL-Python repo, generates documentation, removes unwanted files, and zips it into a zip.
# Removes all folders and files specified by "Doxygen/Files/itemsNotInReleaseVersion.txt". Please add items using their path relative to the top of the RL-Python folder.

# Note: This script is supports Windows and Unix, it does not support OSX. 
# To work, this script needs to call svn, doxygen, and python from the command line. This should not be a problem if python, svn, and doxygen are properly installed.
# It has been tested on Ubuntu and Windows. 

# Please contact gadgy@mit.edu if you have any issues.

try:
    import os, subprocess, shutil, traceback
    #---------------- Name Variables ----------------------#
    REPO_NAME = 'RL-Python'
    OUTPUT_NAME = 'RLPy'
    USER_NAME = 'gadgy'
    DOMAIN_NAME = 'athena.dialup.mit.edu'
    DEST_FOLDER = 'Desktop'

    #---------------- Checkout Repo -----------------------#
    print "Checking Out Fresh Copy of Repo"
    p = subprocess.Popen('svn export svn://acl.mit.edu/acl_collab/agf/' + REPO_NAME, shell = True)
    p.wait()
    
    #---------------- Run Doxygen -------------------------#
    print "Generating Doxygen"
    os.chdir('./' + REPO_NAME + '/Doxygen')
    p = subprocess.Popen('doxygen Doxyfile', shell = True)
    p.wait()

    #---------------- Fix Line Colors ---------------------#
    print "Fixing Colors"
    os.chdir('./Files')
    p = subprocess.Popen('python colorFix.py', shell = True)
    p.wait()
    
    #---------------- Updating Website ---------------#
    print "Updating website"
    p = subprocess.Popen('scp -r ./../Output/testfolder ' + USER_NAME + '@' + DOMAIN_NAME + ':' + DEST_FOLDER, shell = True)
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
    raw_input("success: press enter to exit")
except Exception as e:
    traceback.print_exc()
    raw_input("Error Occured: press enter to exit")

