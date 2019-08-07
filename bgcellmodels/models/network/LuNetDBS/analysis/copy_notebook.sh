#!/bin/bash

usage="
Create notebook_copy2.ipynb:
$ copy_notebook.sh notebook.ipynb 2
"

if [ "$#" -ne 2 ]
then
    echo "Usage: ${usage}"
    exit 1
fi

fullfile=$1
copynumber=$2

filename=$(basename -- "$fullfile")
extension="${filename##*.}"
filename="${filename%.*}"

cp_filename="${filename}_copy${copynumber}"
cp_fullfile="${cp_filename}.${extension}"
cp $fullfile $cp_fullfile

sed -i "s/$filename/$cp_filename/" $cp_fullfile
echo "Created copy ${cp_fullfile}"
