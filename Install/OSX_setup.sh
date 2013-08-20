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
#echo "-------------------------------- CAUTION ----------------------------------"
#echo "THE OSX INSTALLER IS PROVIDED AS A CONVENIENCE, AND LIKELY WILL REQUIRE"
#echo "USER INTERVENTION AT SOME INSTALLATION STEPS."
#echo "If you are uncomfortable modifying system files, we recommend you install RLPy"
#echo "using one of the installers provided for supported operating systems"
#echo "(Windows 32-bit or Ubuntu Linux 32/64-bit),"
#echo "or locate an Apple machine that already has the dependencies installed :)"
#echo "-------------------------------- CAUTION ----------------------------------"
#echo ""
echo ""
echo "Command Line Tools (CLT) and/or XCode are required to use the script."
echo "Ensure you have them installed before proceeding; see \"Install.txt\"."
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
sudo pip install cython # For HIV Domain
sudo pip install networkx # For System Administrator domain

# TODO: Scikit learn also requires setuptools, some sites say no further update req'd.
# Must test below on fresh system. 
sudo pip install -U scikit-learn

# sudo pip install tkinter-pypy

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
    echo -e "We need the absolute location of the RLPy project directory.\n"
    echo -e "It appears to be located in:\n"
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
#        echo -e "\nValid directory specified. "
        echo -e "\n"
#        echo -e "The file RLPy_setup.bash will be created.\n"
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="2"
    fi
done
echo ""
cd $INSTALL_PATH

echo -e "Now configuring cython in setup.py"
python setup.py build_ext --inplace

echo -e "\n"
echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."
