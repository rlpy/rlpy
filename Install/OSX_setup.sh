#!/bin/bash
#VERSION_NUM=$(python --version *.lis|grep -c "")
#
# We modify ~/.launchd.conf since it propagates changes to all programs launched by
# a user, not limited to terminal or GUI apps only.
# See http://stackoverflow.com/questions/603785/environment-variables-in-mac-os-x
#
#TODO - check for installation of XCode
clear
echo "============================= RLPy INSTALLER =============================="
echo "This script installs the required dependencies for the RLPy Framework."
echo "Note that if installation fails, or you wish to install optional packages  "
echo "at any time, you may safely re-run this script.                            "
echo "==========================================================================="
echo ""
echo ""
echo "-------------------------------- CAUTION ----------------------------------"
echo "THE OSX INSTALLER IS PROVIDED AS A CONVENIENCE, AND LIKELY WILL REQUIRE"
echo "USER INTERVENTION AT SOME INSTALLATION STEPS."
echo "If you are uncomfortable modifying system files, we recommend you install RLPy"
echo "using one of the installers provided for supported operating systems"
echo "(Windows 32-bit or Ubuntu Linux 32/64-bit),"
echo "or locate an Apple machine that already has the dependencies installed :)"
echo "-------------------------------- CAUTION ----------------------------------"
echo ""
INVALID_INPUT="1" # Start with improper directory
while [ "$INVALID_INPUT" -ne 0 ]
do
    # pkg-config required for broken matplotlib pip package
    echo "Homebrew is required for installation of pkg-config and other packages."
    echo "Install Homebrew and pkg-config?"
    select yes_no in "Yes" "No";
    do
        case $yes_no in
            Yes ) ruby <(curl -fsSkL raw.github.com/mxcl/homebrew/go); brew doctor; brew install pkg-config; echo -e "\nInstallation of homebrew and pkg-config complete.\n"; INVALID_INPUT="0"; break;;
            No ) echo -e "\nUser opted to ignore homebrew.\n"; INVALID_INPUT="0"; break;;
             * ) echo -e "Unrecognized Input: Please enter [0 or 1]\n\n\n"; break;;
        esac
    done
done

echo Your Python version:
python --version
VALID_VERSION=`python -c 'import sys; print("%i" % (0x020700F0 <= sys.hexversion<0x03000000))'`
if [ $VALID_VERSION -eq 0 ]; then
    echo -e "Please install Python 2.7 <= Version < 3 before proceeding, and ensure"
    echo -e "that you only have one active copy of Python on your PATH variable.\n"
    echo -e "If you suspect competing installations of Python, see http://stackoverflow.com/questions/7746240/in-bash-which-gives-an-incorrect-path-python-versions\n"
    echo -e "To update Python to 2.7 from an earlier version, see http://wolfpaulus.com/journal/mac/installing_python_osx\n"
    echo -e "If you have not yet installed python, we recommend EPD, which includes numpy, scipy, matplotlib, and other requirements"
    echo -e "http://www.enthought.com/products/epd.php\n"
    echo -e "Installation failed."
    exit 0
else
    echo -e "\nPython version valid.\n"
fi

echo -e "Now installing setuptools, required for package installation using pip."
curl -O http://python-distribute.org/distribute_setup.py
sudo python distribute_setup.py

echo -e "\nNow installing gfortran."
brew install gfortran
echo -e "gfortran installation complete."

echo -e "\n Now installing pip."
sudo easy_install pip
echo -e "pip installation complete."

echo -e "\nBeginning installation of required Python packages.\n\n"
#echo -e "Now installing swig, required for scipy."
# nose is used to run unit tests on packages
#sudo pip install nose

#brew install swig
sudo pip install numpy
sudo pip install scipy
brew install pkg-config

# Check for numpy and scipy installation
EXIT_CODE=`python test_numpy_scipy.py`
if [ $EXIT_CODE -eq 1 ]; then
    echo -e "numpy is not installed, or is not recognized by python."
    echo -e "Please install numpy, or see the FAQ if you believe you already have."
    exit 0
fi

if [ $EXIT_CODE -eq 2 ]; then
    echo -e "scipy is not installed, or is not recognized by python."
    echo -e "Please install scipy, or see the FAQ if you believe you already have."
    exit 0
fi

if [ $EXIT_CODE -eq 3 ]; then
    echo -e "neither scipy nor numpy is not installed, or are not recognized by python."
    echo -e "Please install both, or see the FAQ if you believe you already have."
    exit 0
fi
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "Both numpy and scipy have been detected.  Installation will proceed.\n"
fi

sudo pip install matplotlib
sudo pip install networkx

# TODO: Scikit learn also requires setuptools, some sites say no further update req'd.
# Must test below on fresh system. 
sudo pip install -U scikit-learn

echo -e "A list of all installed packages is shown below.\n"
pip freeze

echo ""
echo "You should see a list which contains the following at a minimum-"
echo "any missing packages likely did not install correctly:"
echo ""

echo "matplotlib==1.2.0"
echo "networkx==1.7"
echo "numpy==1.6.2"
echo "scikit-learn==0.12.1"
echo "scipy==0.11.0"

echo ""

read -p "Ready to continue to final step of installation? (Press [Enter])"
clear

# Final step adds a line to .launchd.conf to source the file RLPy_setup.bash,
# after the user locates the install directory.
# The file RLPy_setup.bash is included in the repository, but we have
# commented code here to automatically regenerate it.
#
# TODO Test: if the environment variable is not properly set after testing,
# modify a plist instead: http://stackoverflow.com/questions/7501678/set-environment-variables-on-mac-os-x-lion#

VALID_DIRECTORY_ZERO="1" # Start with improper directory
while [ "$VALID_DIRECTORY_ZERO" -ne 0 ]
do
    # Set the environment variable RLPy_ROOT
    echo -e "We need to set an environment variable for the location of the RLPy"
    echo -e "project directory.\nIt appears to be located in:\n"
    cd ..
    pwd
    cd - > /dev/null
    # Above suppresses output of cd - command, which returns to previous directory.

    echo -e "\nIs this correct? (And where you intend to continue working from?)"
    echo -e "[Enter 1 or 2]"
    echo -e "1) Yes"
    echo -e "2) No"
    read yes_no
    INSTALL_PATH=""
    case $yes_no in
        1) cd ..
             INSTALL_PATH=`pwd`
             cd - > /dev/null; VALID_DIRECTORY_ZERO="0"
             ;;
        2)  echo -e "Please enter the absolute path to the RLPy root directory: "
            # Change to root directory in case a sneaky user tries to specify
            # a relative path
             cd /
             read INSTALL_PATH
             cd $INSTALL_PATH
             VALID_DIRECTORY_ZERO=$?
             ;;
        *)  echo -e "\nUnrecognized Input: Please enter [0 or 1].\n\n\n"
            continue
             ;;
    esac
    if [ $VALID_DIRECTORY_ZERO -eq 0 ]; then
        echo -e "\nValid directory specified. "
#        echo -e "The file RLPy_setup.bash will be created.\n"
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="2"
    fi
done
echo ""
cd $INSTALL_PATH
#echo "Creating file RLPy_setup.bash in directory $INSTALL_PATH"
# sudo echo 'export RLPy_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" ' > RLPy_setup.bash

# Make a backup of the .launchd.conf file before editing!
cd ~
HOMEDIR=`pwd`

# Only copy the .launchd.conf file if it exists (to prevent throwing an error)
if [ -f .launchd.conf  ]; then
    sudo cp .launchd.conf .launchd.conf_RLPy_BACKUP
fi
# Determine if .launchd.conf already sources the RLPy_setup file in some way
ALREADY_EXPORTED=`find $HOMEDIR -name '.launchd.conf' -maxdepth 1 -exec grep RLPy_setup.bash {} +`

# A file is already sourced from .launchd.conf
if [ "$?" -eq 0 ]; then
    echo -e "Your .launchd.conf file already appears to source RLPy_setup.bash;"
    echo -e "this line will be overwritten with the newly created one based on"
    echo -e "your answer above."
    # Delete the line(s) containing 'RLPy_setup.bash'
    # NOTE THE DIFFERENT SYNTAX for 'sed' on OSX, need to explicitly include backup
    # extension for the file, .BAK - see 
    # https://sites.google.com/site/randomprogrammingnotes/macosxsed
    sudo sed -i .BAK '/RLPy_setup.bash/d' .launchd.conf > /dev/null
fi

echo -e "\nAdding source of RLPy_setup.bash to .launchd.conf\n"

#if [ "$?" -eq 0 ]; then
    sudo echo "# Automatically added RLPy_setup.bash below by OSX_setup.sh script for RLPy" >> .launchd.conf
    sudo echo "source $INSTALL_PATH/RLPy_setup.bash" >> .launchd.conf
    echo -e "Successfully modified .launchd.conf\n"
#else
#    echo -e "There was a problem creating a backup of .launchd.conf.  You will need \n"
#    echo -e "to manually 'source' the file RLPy_setup.bash for your shell\n"
#    echo -e "environment; we recommend adding it to whatever startup script\n"
#    echo -e "is used on your machine.\n"
#fi



# Finally, create the config.py file which we can use to add other environment
# variables to our project.

VALID_DIRECTORY_ZERO="1" # Start with improper directory
while [ "$VALID_DIRECTORY_ZERO" -ne 0 ]
do
    echo -e "\n"
    echo -e "Final step:"
    echo -e "Please enter a directory in which to store matplotlib temporary"
    echo -e "files; the only constraint is that you have read/write priveleges to"
    echo -e "this directory."
    echo -e ""

    echo -e "May we suggest: $HOMEDIR/mpl_tmp ."
    echo -e "Is this ok? [Enter 1 or 2]"
    echo -e "1) Yes"
    echo -e "2) No"
    read yes_no
    TMP_PATH=""
    case $yes_no in
        1) TMP_PATH="$HOMEDIR/mpl_tmp"
             ;;
        2)  echo -e "Please enter the absolute path to a temporary directory of choice: "
            # Change to root directory in case a sneaky user tries to specify
            # a relative path
             cd /
             read TMP_PATH
             ;;
         *)  echo -e "Unrecognized Input: Please enter [0 or 1].\n\n\n"
             continue
             ;;
    esac
    #-p option makes directories only as needed.
    mkdir -p $TMP_PATH
    VALID_DIRECTORY_ZERO="$?"
    if [ $VALID_DIRECTORY_ZERO -eq 0 ]; then
        echo -e "\nValid directory specified. This may take some time."
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="2"
    fi
done
echo ""

# Now create the config.py file
cd $INSTALL_PATH
(
echo "# *****************************************************************************"
echo "# *** WARNING: CHANGES TO THIS FILE WILL BE OVERWRITTEN BY INSTALLER SCRIPT ***"
echo "# *****************************************************************************"
echo ""
echo "HOME_DIR = r\"$TMP_PATH\" # Where to store tempfiles for matplotlib"
echo "CONDOR_CLUSTER_PREFIX = '/data' # not used anywhere as part of path,"
echo "# only as unique identifier distinguishing cluster from normal local machine."
echo "# See isOnCluster()."
) > Config.py

# No need to create desktop shortcut to source files, .launchd.conf automatically
# adds variables on user login to all programs launched by user.

# Below removed per agf request, but it remains a true statement :)
# As of this writing, we just aren't relying on environment variables at all.
# TODO - Suggest exporting the variable at end of script, in addition to writing
# config file, to eliminate need for restart.


#echo -e "You must *** RESTART YOUR COMPUTER *** for environmental"
#echo -e "variable changes to take effect."


echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."
