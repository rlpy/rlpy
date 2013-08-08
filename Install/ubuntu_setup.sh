#!/bin/bash
#VERSION_NUM=$(python --version *.lis|grep -c "")
clear
echo "============================== RLPy INSTALLER ============================="
echo "This script installs the required dependencies for the RLPy Framework."
echo "Note that if installation fails, or you wish to install optional packages  "
echo "at any time, you may safely re-run this script.                            "
echo "==========================================================================="
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

echo -e "\nBeginning installation of required packages.\n\n"
sudo apt-get install python-dev python-setuptools python-numpy python-scipy python-matplotlib python-networkx graphviz
INVALID_INPUT="1" # Start with improper directory
while [ "$INVALID_INPUT" -ne 0 ]
do
        echo -e "\nDo you want to install the package scikit-learn as well?"
        echo -e "(Highly recommended, required for Pendulum domain and BEBF representation)"
        echo -e "[Enter 1 or 2]"
        select yes_no in "Yes" "No";
        do
            case $yes_no in
                Yes ) sudo apt-get install libatlas-dev gfortran python-pip; sudo pip install -U scikit-learn; echo -e "\nInstallation of scikit-learn complete.\n"; INVALID_INPUT="0"; break;;
                No )  echo -e "\nUser opted to ignore scikit-learn.\n"; INVALID_INPUT="0"; break;;
                *)    echo -e "Unrecognized Input: Please enter [0 or 1].\n\n\n"; break;;
            esac
        done
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
# to source the file RLPy_setup.bash, after the user locates the install directory.
# The file RLPy_setup.bash is included in the repository, but we have
# commented code here to automatically regenerate it.
VALID_DIRECTORY_ZERO="1" # Start with improper directory
while [ "$VALID_DIRECTORY_ZERO" -ne 0 ]
do
    # Set the environment variable RL_PYTHON_ROOT
    echo -e "We need the location of the RLPy project directory.\n"
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
             read INSTALL_PATH
             cd $INSTALL_PATH
             VALID_DIRECTORY_ZERO=$?
             ;;
        *)  echo -e "\nUnrecognized Input: Please enter [0 or 1].\n\n\n"
            continue
             ;;
    esac
    if [ $VALID_DIRECTORY_ZERO -eq 0 ]; then
        echo -e "\nValid directory specified. This may take some time."
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

# Make a backup of the .bashrc and environment files before editing!
cd
HOMEDIR=`pwd`

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

    echo -e "May we suggest: $HOMEDIR/mpl_tmp "
    echo -e "Is this ok? [Enter 1 or 2]"
    echo -e "1) Yes"
    echo -e "2) No"
    read yes_no
    TMP_PATH=""
    case $yes_no in
        1)  TMP_PATH="$HOMEDIR/mpl_tmp"
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
        echo -e "\nValid directory specified. "
    else
        echo -e "\nYou specified an invalid directory; maybe you haven't created it yet?\n"
        # Automatically force entry of python path in loop above
        yes_no="2"
    fi
done
echo ""

INVALID_INPUT="1" # Start with improper directory
while [ "$INVALID_INPUT" -ne 0 ]
do
    # Create shortcut on desktop to automatically source files
    echo -e "\nLastly, would you like a shortcut to be created on your desktop which will"
    echo -e "automatically source the required files on eclipse startup?"
    echo -e "Note that we assume a single default eclipse installation."
    echo -e "See (http://answers.ros.org/question/29424/eclipse-ros-fuerte/)"
    echo -e "to create a custom shortcut."
    echo -e "[Enter 1 or 2] :"
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
                  ) > "$HOMEDIR/Desktop/RLPy_Eclipse_Env.Desktop"
                  echo -e "\n\n"
                  echo -e "*******************************************************************************"
                  echo -e "You may need to right-click the icon, go to properties->permissions,"
                  echo -e "and check the box which enables execution."
                  INVALID_INPUT="0"
                  break;;
            No ) echo -e "\n\n"
                 echo -e "*******************************************************************************"
#                 echo -e "Without this shortcut, you have four options to obtain necessary variables:"
#                 echo -e "1) source ~/.bashrc and/or /etc/environment whenever launching eclipse"
#                 echo -e "2) Launch eclipse from the console, so that it receives needed variables."
#                 echo -e "3) Create a custom shortcut - see:"
#                 echo -e "[http://answers.ros.org/question/29424/eclipse-ros-fuerte/]"
#                 echo -e "4) Add the RL_Python_ROOT variable to your RLPy Eclipse project in:"
                 echo -e "Without this shortcut, the easiest way to obtain necessary environment"
                 echo -e "variables is to add it to your IDE project directly.  In Eclipse:"
                 echo -e "window->preferences->pydev->interpreter Pydev->environment"
                 echo -e "Create the variable RL_Python_ROOT and set it accordingly."
                 echo -e ""
                 echo -e "FYI, earlier in the installation, you chose RL_Python_ROOT ="
                 echo -e "$INSTALL_PATH"
                 INVALID_INPUT="0"
                 break;;
             * ) echo -e "Unrecognized Input: Please enter [0 or 1].\n\n\n"
                 break;;
        esac
    done
done
echo -e "\n"

echo -e "\n"
read -p "Installation script complete, press [Enter] to exit."

