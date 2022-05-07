#!/bin/bash

declare -a data_array=(0001 0002 0005 0009 0011 0013 0014 0015 0017 0018 0019 0020 0022 0023 0027 0028 0029 0032 0035 0036 0039 0046 0048 0051 0052 0056 0057 0059 0060 0061 0064 0070 0079 0084 0086 0087 0091 0093 0095 0096 0101 0104 0106 0113 0117)

process()
{
    wget -k -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_$1/2011_09_26_drive_$1_extract.zip
    wget -k -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_$1/2011_09_26_drive_$1_sync.zip
    echo "Downloading $1"

    unzip 2011_09_26_drive_$1_extract.zip
    unzip 2011_09_26_drive_$1_sync.zip
    echo "Extracting $1"

    echo "Delete Zip files"
    rm -rf 2011_09_26_*.zip
}

download_calib(){
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
    unzip 2011_09_26_calib.zip
    rm -rf 2011_09_26_*.zip
}
# get length of an array
data_array_length=${#data_array[@]}

echo "Downloading ${data_array_length} data from KITTI..."

for (( i=1; i<${data_array_length}+1; i++ ));
do
  echo $i " / " ${data_array_length} " : " ${data_array[$i-1]} && wait
  process ${data_array[i-1]} &
done

download_calib

wait
echo "All processes done!"