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


# TODO: Install / check for installation of Tex fonts on OSX, if possible

echo -e "Now installing setuptools, required for package installation using pip."
curl -O http://python-distribute.org/distribute_setup.py
sudo python distribute_setup.py

echo -e "\n Now installing pip"
sudo easy_install pip
echo -e "pip installation complete"

echo -e "\nBeginning installation of required packages.\n\n"
echo -e "Now installing swig, required for scipy"
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
echo "Note that this script does not install required latex files"
echo ""
echo "Installation script complete."
