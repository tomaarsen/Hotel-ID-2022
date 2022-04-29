if [ $# -ne 2 ]; then
    echo "Please supply a path to the best and last model to save."
    exit 1
fi

best=$1
last=$2

# Save the current changes that might exist on main
git stash

# Move to the models branch
git checkout models

# Update the models
rm models/*.ckpt
cp $best models/
cp $last models/
git add models/
git commit -m "Save new models & clear previous ones"

# Move back to main
git checkout main

# Restore last changes in the stash
git stash pop