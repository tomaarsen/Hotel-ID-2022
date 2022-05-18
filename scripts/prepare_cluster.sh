#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
CEPH_USER_DIR=/ceph/csedu-scratch/course/IMC030_MLIP/users/"$USER"
DATA_DIR=/scratch/"$USER"/data/hotel-2022
mkdir -p "$CEPH_USER_DIR"/slurm
chmod 700 "$CEPH_USER_DIR" # only you can access
ln -sfn "$CEPH_USER_DIR" "$SCRIPT_DIR"/../logs

# # place `~/.cache`, and optionally `~/.local`, in the ceph user directory in order to
# # save disk space in $HOME folder
# function setup_link {
#   dest_path=$1
#   link_path=$2

#   if [ -L "${link_path}" ] ; then
#     # link_path exists as a link
#     if [ -e "${link_path}" ] ; then
#       # and works
#       echo "link at $link_path is already setup"
#     else
#       # but is broken
#       echo "link $link_path is broken... Does $dest_path exists?"
#       return 1
#     fi
#   elif [ -e "${link_path}" ] ; then
#     # link_path exists, but is not a link
#     mkdir -p "$dest_path"
#     echo "moving all data in $link_path to $dest_path"
#     mv "$link_path"/* "$dest_path"/
#     rmdir "$link_path"
#     ln -s "$dest_path" "$link_path"
#     echo "created link $link_path to $dest_path"
#   else
#     # link_path does not exist
#     mkdir -p "$dest_path"
#     ln -s "$dest_path" "$link_path"

#     echo "created link $link_path to $dest_path"
#   fi

#   return 0
# }

# # .local is probably not necessary
# setup_link "$CEPH_USER_DIR"/.local ~/.local
# setup_link "$CEPH_USER_DIR"/.cache ~/.cache


# install the `virtualenv` command
python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv

# set up a virtual environment located at
# /scratch/$USER/virtual_environments/tiny-voxceleb-venv
# and make a symlink to the virtual environment
# at the root directory of this project called "venv"
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN99 ###"
./setup_virtual_environment.sh

# make sure that there's also a virtual environment
# on the GPU nodes
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
ssh cn47 "
  source .profile
  cd $PWD;
  ./setup_virtual_environment.sh
"

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
ssh cn48 "
  source .profile
  cd $PWD;
  ./setup_virtual_environment.sh
"

# make a symlink to the data in order to directly access it from the root of the project
mkdir -p "$SCRIPT_DIR"/../data
ln -sfn /ceph/csedu-scratch/course/IMC030_MLIP/users/data_task2 "$SCRIPT_DIR"/../data