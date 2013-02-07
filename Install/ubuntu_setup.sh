#!/bin/bash
#VERSION_NUM=$(python --version *.lis|grep -c "")
clear
echo "This script installs the required dependencies for the RL_Python Framework."
echo "Note that if installation fails, or you wish to install optional packages"
echo "at any time, you may safely re-run this script."
echo ""

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



echo "Do you wish to install latex fonts as well, required for some visualizations?"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) sudo apt-get install texlive-latex-extra texlive-fonts-recommended; echo -e "\nInstallation of Tex fonts Complete.\n"; break;;
        No ) echo -e "\nUser opted to ignore addons.\n"; break;;
    esac
done

echo -e "\nBeginning installation of required packages.\n\n"
sudo apt-get install python-dev python-setuptools python-numpy python-scipy python-matplotlib python-networkx graphviz

echo -e "\nDo you want to install the package scikit-learn as well?"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) sudo apt-get install libatlas-dev gfortran python-pip; sudo pip install -U scikit-learn; echo -e "\nInstallation of scikit-learn complete.\n"; break;;
        No ) echo -e "\nUser opted to ignore scikit-learn.\n"; break;;
    esac
done
echo ""

echo "The status of optional packages is shown below:"
dpkg -s texlive-latex-extra |  grep Status
dpkg -s texlive-fonts-recommended |  grep Status
dpkg -s libatlas-dev |  grep Status

echo -e "\n\nThe status of all required packages are shown below:"
dpkg -s python-dev |  grep Status
dpkg -s python-setuptools |  grep Status
dpkg -s python-numpy |  grep Status
dpkg -s python-scipy |  grep Status
dpkg -s python-matplotlib |  grep Status
dpkg -s graphviz |  grep Status
dpkg -s python-networkx |  grep Status

echo -e "\n\nInstallation script complete."
