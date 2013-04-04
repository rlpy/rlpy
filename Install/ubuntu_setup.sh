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

# Final step adds a line to .bashrc and /etc/environment
# to source the file RL_Python_setup.bash, after the user locates the install directory.
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

# Make a backup of the .bashrc and environment files before editing!
cd
HOMEDIR=`pwd`

sudo cp .bashrc .bashrc_RL_PYTHON_BACKUP
sudo cp /etc/environment /etc/environment_RL_PYTHON_BACKUP

# Determine if .bashrc already sources this file in some way
ALREADY_EXPORTED=`find $HOMEDIR -name '.bashrc' -exec grep RL_Python_setup.bash {} +`

# A file is already sourced from bashrc
if [ "$?" -eq 0 ]; then
    echo -e "Your .bashrc file already appears to source RL_Python_setup.bash;"
    echo -e "this line will be overwritten with the newly created one based on"
    echo -e "your answer above."
    # Delete the line(s) containing 'RL_Python_setup.bash'
    sudo sed -i '/RL_Python_setup.bash/d' .bashrc > /dev/null
fi

# Determine if /etc/environment already sources this file in some way
ALREADY_EXPORTED=`sudo find /etc -name 'environment' -exec grep RL_Python_setup.bash {} +`

# A file is already sourced from environment
if [ "$?" -eq 0 ]; then
    cd /etc
    echo -e "Your /etc/environment file already appears to source RL_Python_setup.bash;"
    echo -e "this line will be overwritten with the newly created one based on"
    echo -e "your answer above."
    # Delete the line(s) containing 'RL_Python_setup.bash'
    sudo sed -i '/RL_Python_setup.bash/d' environment > /dev/null
fi

echo -e "\nAdding source of RL_Python_setup.bash to environment ..."

cd /etc

# Note we must use sudo su (or execute as root directly) in order to modify
# the /etc/environment file [apparently... sudo echo etc. does not work].
# Thus the crazy syntax below.

#if [ "$?" -eq 0 ]; then
    sudo -u root -H sh -c "echo '# Automatically added RL_Python_setup.bash below by ubuntu_setup.sh script for RL-Python' >> /etc/environment"
    sudo -u root -H sh -c "echo 'source $INSTALL_PATH/RL_Python_setup.bash' >> /etc/environment"
    echo -e "Successfully modified environment.\n"

echo -e "Adding source of RL_Python_setup.bash to .bashrc ..."

cd $HOMEDIR
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


# Finally, create the config.py file which we can use to add other environment
# variables to our project.

echo -e "\n"
echo -e "Final step:"
echo -e "Please enter a directory in which to store matplotlib temporary"
echo -e "files; the only constraint is that you have read/write priveleges to"
echo -e "this directory."
echo -e ""

echo -e "May we suggest: $HOMEDIR/mpl_tmp ."
echo -e "Is this ok? (Yes or No)"
read yes_no
TMP_PATH="null"
VALID_DIRECTORY_ZERO="1" # Start with improper directory
while [ "$VALID_DIRECTORY_ZERO" -ne 0 ]
do
    case $yes_no in
        Yes) TMP_PATH="$HOMEDIR/mpl_tmp"
             ;;
        No)  echo -e "Please enter the absolute path to a temporary directory of choice: "
             read TMP_PATH
             ;;
    esac
    #-p option makes directories only as needed.
    mkdir -p $TMP_PATH
    VALID_DIRECTORY_ZERO="$?"
    if [ $VALID_DIRECTORY_ZERO -eq 0 ]; then
        echo -e "\nValid directory specified. "
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="No"
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

# Create shortcut on desktop to automatically source files
echo -e "\nLastly, would you like a shortcut to be created on your desktop which will"
echo -e "automatically source the required files on eclipse startup?"
echo -e "Note that we assume a single default eclipse installation."
echo -e "See (http://answers.ros.org/question/29424/eclipse-ros-fuerte/)"
echo -e "to create a custom shortcut."
echo -e "Select (Yes, No) :"
select yes_no in "Yes" "No";
do
    case $yes_no in
        Yes ) (
                echo -e ""
                echo -e "[Desktop Entry]"
                echo -e "Version=1.0"
                echo -e "Type=Application"
                echo -e "Terminal=false"
                echo -e "Icon[en_US]=/opt/eclipse/icon.xpm"
                echo -e "Exec=bash -c \"source ~/.bashrc; source /etc/environment; /opt/eclipse/eclipse\""
                echo -e "Name[en_US]=Eclipse"
                echo -e "Name=Eclipse"
                echo -e "Icon=/opt/eclipse/icon.xpm"
              ) > "$HOMEDIR/Desktop/RL_Python_Eclipse_Env.Desktop"
              echo -e "\n\nYou may need to right-click the icon, go to properties->permissions,"
              echo -e "and check the box which enables execution."
              break;;
        No ) echo -e "\n\nYou will need to source ~/.bashrc and/or /etc/environment"
             echo -e "for necessary environmental variables to be available to eclipse."
             echo -e "Alternatively, launch eclipse from the console."
             break;;
    esac
done
echo ""

echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."
