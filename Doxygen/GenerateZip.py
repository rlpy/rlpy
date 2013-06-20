#!/usr/bin/env python

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
    
    USER_NAME = 'acl'
    DOMAIN_NAME = 'acl.mit.edu'
    DEST_FOLDER = '/var/www/acl.mit.edu/htdocs/'
    #OPTIONS = '-rvq' VERBOS
    OPTIONS = '-rq'
    

    #---------------- Remove Files From Previous Runs -----------------------#
    for dir in ['RLPy', 'RL-Python','Output']:
        if os.path.exists(dir): shutil.rmtree(dir)
    if os.path.isfile('RLPy.zip'): os.remove('RLPy.Zip')

    #---------------- Checkout Repo -----------------------#
    print "Checking Out Fresh Copy of Repo"
    p = subprocess.Popen('svn -q export svn://acl.mit.edu/acl_collab/agf/' + REPO_NAME, shell = True)
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
    #if os.path.exist('./' +REPO_NAME):
    os.rename('./' +REPO_NAME,'./' +OUTPUT_NAME)
    shutil.make_archive(OUTPUT_NAME, "zip", root_dir='.', base_dir='.//' + OUTPUT_NAME)
    shutil.copy('./' +OUTPUT_NAME + '.zip', './' + OUTPUT_NAME+ '/Doxygen/Output/' + OUTPUT_NAME)

    #---------------- Updating Website ---------------#
    print "Updating website"
    p = subprocess.Popen('scp '+ OPTIONS +' ./' +OUTPUT_NAME +'/Doxygen/Output/RLPy ' + USER_NAME + '@' + DOMAIN_NAME + ':' + DEST_FOLDER, shell = True)
    p.wait()
    #print "THIS IS THE COMMAND THAT I WOULD USE TO SCP"
    #print 'scp '+ OPTIONS +' ./' +OUTPUT_NAME +'/Doxygen/Output/RLPy ' + USER_NAME + '@' + DOMAIN_NAME + ':' + DEST_FOLDER
    #print 'Note that the destination folder is /var/www/acl.mit.edu/htdocs/ not /var/www/acl.mit.edu/htdocs/RLPy.'
    #print 'This is because I am scping the entire RLPy folder; the former address will enable the new folder to overwright the old while the later would create a new RLPy INSIDE the old RLPy'
    #print 
    
    #---------------- Remove Checkout ----------------------#
    print "Removing Temporary Checkout"
    shutil.rmtree('./' +OUTPUT_NAME)
    raw_input("success: press enter to exit")
except Exception as e:
    traceback.print_exc()
    raw_input("Error Occured: press enter to exit")

