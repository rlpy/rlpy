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
if [ "$VALID_VERSION" -eq 0 ]; then
    echo -e "Please install Python 2.7 <= Version < 3 before proceeding, and ensure"
    echo -e "that you only have one active copy of Python on your PATH variable.\n"
    echo -e "If you suspect competing installations of Python, see http://stackoverflow.com/questions/7746240/in-bash-which-gives-an-incorrect-path-python-versions\n"
    echo -e "Installation failed."
    exit 0
else
    echo -e "\nPython version valid.\n"
fi


# No longer require latex fonts, matplotlib is capable of displaying equations.
#
#echo "Do you wish to install latex fonts as well, required for some visualizations?"
#select yes_no in "Yes" "No";
#do
#    case $yes_no in
#        Yes ) sudo apt-get install texlive-latex-extra texlive-fonts-recommended; echo -e #"\nInstallation of Tex fonts Complete.\n"; break;;
#        No ) echo -e "\nUser opted to ignore addons.\n"; break;;
#    esac
#done

echo -e "\nBeginning installation of required packages.\n\n"
sudo apt-get install python-dev python-setuptools python-numpy python-scipy python-matplotlib python-networkx graphviz

echo -e "\nDo you want to install the package scikit-learn as well? (Highly recommended, required for Pendulum domain and BEBF representation)"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) sudo apt-get install libatlas-dev gfortran python-pip; sudo pip install -U scikit-learn; echo -e "\nInstallation of scikit-learn complete.\n"; break;;
        No ) echo -e "\nUser opted to ignore scikit-learn.\n"; break;;
    esac
done
echo ""

echo "The status of optional packages is shown below:"
#dpkg -s texlive-latex-extra |  grep Status
#dpkg -s texlive-fonts-recommended |  grep Status
dpkg -s libatlas-dev |  grep 'Package\|Status';

echo -e "\n\nThe statuses of all required packages are shown below:"
dpkg -s python-dev |  grep 'Package\|Status';
dpkg -s python-setuptools |  grep 'Package\|Status';
dpkg -s python-numpy |  grep 'Package\|Status';
dpkg -s python-scipy |  grep 'Package\|Status';
dpkg -s python-matplotlib |  grep 'Package\|Status';
dpkg -s graphviz |  grep 'Package\|Status';
dpkg -s python-networkx |  grep 'Package\|Status';

echo -e "\nIf any required packages are shown as missing, try re-running this script;"
echo -e "otherwise attempt sudo apt-get install <<missing package>> manually.\n"

read -p "Ready to continue to final step of installation? (Press [Enter])"
clear

# Final step adds a line to .bashrc to source the file RL_Python_setup.bash,
# after the user locates the install directory.
# The file RL_Python_setup.bash is included in the repository, but we have
# commented code here to automatically regenerate it.

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
cd
sudo cp .bashrc .bashrc_RL_PYTHON_BACKUP

# Determine if .bashrc already sources this file in some way
ALREADY_EXPORTED=`find ~ -name '.bashrc' -exec grep RL_Python_setup.bash {} +`

# A file is already sourced from bashrc
if [ "$?" -eq 0 ]; then
    echo -e "Your .bashrc file already appears to source RL_Python_setup.bash;"
    echo -e "this line will be overwritten with the newly created one based on"
    echo -e "your answer above."
    # Delete the line(s) containing 'RL_Python_setup.bash'
    sed -i '/RL_Python_setup.bash/d' .bashrc > /dev/null
fi

echo -e "\nAdding source of RL_Python_setup.bash to .bashrc\n"

#if [ "$?" -eq 0 ]; then
    sudo echo "# Automatically added RL_Python_setup.bash below by ubuntu_setup.sh script for RL-Python" >> .bashrc
    sudo echo "source $INSTALL_PATH/RL_Python_setup.bash" >> .bashrc
    echo -e "Successfully modified .bashrc\n"
#else
#    echo -e "There was a problem creating a backup of .bashrc.  You will need \n"
#    echo -e "to manually 'source' the file RL_Python_setup.bash for your shell\n"
#    echo -e "environment; we recommend adding it to whatever startup script\n"
#    echo -e "is used on your machine.\n"
#fi


echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."
