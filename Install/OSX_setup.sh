#!/bin/bash
#VERSION_NUM=$(python --version *.lis|grep -c "")
#
#TODO - check for installation of XCode
clear
echo "This script installs the required dependencies for the RL_Python Framework."
echo "Note that if installation fails, or you wish to install optional packages"
echo "at any time, you may safely re-run this script."
echo ""
# pkg-config required for broken matplotlib pip package
echo "Homebrew is required for installation of pkg-config and other packages.  Install Homebrew and pkg-config?"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) ruby <(curl -fsSkL raw.github.com/mxcl/homebrew/go); brew doctor; brew install pkg-config; echo -e "\nInstallation of homebrew and pkg-config complete.\n"; break;;
        No ) echo -e "\nUser opted to ignore homebrew.\n"; break;;
    esac
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

echo -e "Now installing setuptools, required for package installation using pip."
curl -O http://python-distribute.org/distribute_setup.py
sudo python distribute_setup.py

echo -e "\n Now installing pip."
sudo easy_install pip
echo -e "pip installation complete."

echo -e "\nBeginning installation of required packages.\n\n"
echo -e "Now installing swig, required for scipy."
brew install swig
#sudo pip install numpy
#sudo pip install scipy
#sudo pip install matplotlib
sudo pip install networkx
# nose is used to run unit tests on packages
#sudo pip install nose

# TODO: Scikit learn also requires setuptools, some sites say no further update req'd.
# Must test below on fresh system. 
echo -e "\nDo you want to install the package scikit-learn as well?"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) sudo pip install -U scikit-learn; echo -e "\nInstallation of scikit-learn complete.\n"; break;;
        No ) echo -e "\nUser opted to ignore scikit-learn.\n"; break;;
    esac
done

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

# Final step adds a line to .bash_profile to source the file RL_Python_setup.bash,
# after the user locates the install directory.
# The file RL_Python_setup.bash is included in the repository, but we have
# commented code here to automatically regenerate it.
#
# TODO Test: if the environment variable is not properly set after testing,
# modify a plist instead: http://stackoverflow.com/questions/7501678/set-environment-variables-on-mac-os-x-lion#

# Set the environment variable RL_PYTHON_ROOT
echo -e "We need to set an environment variable for the location of the RL-Python"
echo -e "project directory.\nIt appears to be located in:\n"
cd ..
pwd
cd - > /dev/null
# Above suppresses output of cd - command, which returns to previous directory.

echo -e "\nIs this correct? (And where you intend to continue working from?)\n [Yes or No]"
read yes_no
INSTALL_PATH="null"
VALID_DIRECTORY_ZERO="1" # Start with improper directory
while [ "$VALID_DIRECTORY_ZERO" -ne 0 ]
do
    case $yes_no in
        Yes) cd ..
             INSTALL_PATH=`pwd`
             cd - > /dev/null; VALID_DIRECTORY_ZERO="0"
             ;;
        No)  echo -e "Please enter the absolute path to the RL-Python root directory: "
             read INSTALL_PATH
             cd $INSTALL_PATH
             VALID_DIRECTORY_ZERO=$?
             ;;
    esac
    if [ $VALID_DIRECTORY_ZERO -eq 0 ]; then
        echo -e "\nValid directory specified. "
#        echo -e "The file RL_Python_setup.bash will be created.\n"
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="No"
    fi
done
echo ""
cd $INSTALL_PATH
#echo "Creating file RL_Python_setup.bash in directory $INSTALL_PATH"
# sudo echo 'export RL_PYTHON_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" ' > RL_Python_setup.bash

# Make a backup of the .bashrc file before editing!
cd ~
sudo cp .bash_profile .bash_profile_RL_PYTHON_BACKUP

# Determine if .bashrc already sources this file in some way
ALREADY_EXPORTED=`find ~ -name '.bash_profile' -exec grep RL_Python_setup.bash {} +`

# A file is already sourced from bash_profile
if [ "$?" -eq 0 ]; then
    echo -e "Your .bash_profile file already appears to source RL_Python_setup.bash;"
    echo -e "this line will be overwritten with the newly created one based on"
    echo -e "your answer above."
    # Delete the line(s) containing 'RL_Python_setup.bash'
    sed -i '/RL_Python_setup.bash/d' .bash_profile > /dev/null
fi

echo -e "\nAdding source of RL_Python_setup.bash to .bash_profile\n"

#if [ "$?" -eq 0 ]; then
    sudo echo "# Automatically added RL_Python_setup.bash below by OSX_setup.sh script for RL-Python" >> .bash_profile
    sudo echo "source $INSTALL_PATH/RL_Python_setup.bash" >> .bash_profile
    echo -e "Successfully modified .bash_profile\n"
#else
#    echo -e "There was a problem creating a backup of .bash_profile.  You will need \n"
#    echo -e "to manually 'source' the file RL_Python_setup.bash for your shell\n"
#    echo -e "environment; we recommend adding it to whatever startup script\n"
#    echo -e "is used on your machine.\n"
#fi

echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."
