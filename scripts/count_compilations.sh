#!/bin/sh

# currently only works for autoencoded examples
EXAMPLES_DIR=$1

if [ -z "$EXAMPLES_DIR" ]
then
    echo 'No example directory given'
    exit 1
fi


# function compile_check {
compile_check () {
    FILEPATH=$1

    if [ -z "$FILEPATH" ]
    then
        echo 'No example directory given'
        exit 1
    fi

    ghc -fno-code $FILEPATH > /dev/null 2> /dev/null
    if [[ $? == 0 ]]; then
        echo ${FILEPATH}
    fi
}
export -f compile_check

(ls $EXAMPLES_DIR/autoencoded | parallel --joblog log.txt --progress --bar compile_check $EXAMPLES_DIR/autoencoded/{}/autoencoded_code.hs) > $EXAMPLES_DIR/autoencoded/compiled_examples.txt

NUM_COMPILED=`cat $EXAMPLES_DIR/autoencoded/compiled_examples.txt | wc -l`

echo "$NUM_COMPILED autoencoded examples compiled"
