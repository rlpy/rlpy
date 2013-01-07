#!/bin/bash
#VERSION_NUM=$(python --version *.lis|grep -c "")
#
# On OSX, it might be necessary to do a check for XCode, development environment for mac
clear
echo "This script installs the required dependencies for the RL_Python Framework."
echo "Note that if installation fails, or you wish to install optional packages"
echo "at any time, you may safely re-run this script."
echo ""
# pkg-config required for broken matplotlib pip package
echo "Homebrew is required for installation of fortran compiler and pkg-config.  Install Homebrew, gfortran, pkg-config?"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) ruby <(curl -fsSkL raw.github.com/mxcl/homebrew/go); brew install gfortran; brew install pkg-config; echo -e "\nInstallation of homebrew and gfortran complete.\n"; break;;
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
    echo -e "Installation failed."
    exit 0
else
    echo -e "\nPython version valid.\n"
fi

# TODO: Install / check for installation of Tex fonts on OSX, if possible

echo -e "\n Now installing pip"
brew install pip
echo -e "pip installation complete"

echo -e "\nBeginning installation of required packages.\n\n"
sudo pip install numpy
sudo pip install scipy
sudo pip install matplotlib
sudo pip install networkx


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

echo "A list of all installed packages is shown below."
pip freeze

echo ""
echo "Installation script complete."
