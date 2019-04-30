#!/bin/bash

# http://frostt.io/tensors/

get_dataset() {
    URL=$1
    GZ_FILENAME=${URL##*/}
    FILENAME=${GZ_FILENAME%%.gz*}

    if [ -f datasets/$FILENAME ]; then
        echo "dataset $FILENAME already downloaded"
    else
        cd datasets
        echo Downloading $GZ_FILENAME
        wget $URL
        echo Unzipping to $FILENAME ...
        gunzip $GZ_FILENAME
        cd -
    fi
}

if [ ! -d datasets/ ]; then
    mkdir datasets/
fi

# order-3 tensors
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
#get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_2-2-2.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_3-3-3.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_4-3-2.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_4-4-3.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_4-4-4.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_5-5-5.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_6-3-3.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-3d.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-3d.tns.gz

# higher-order tensors
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz
get_dataset https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz
