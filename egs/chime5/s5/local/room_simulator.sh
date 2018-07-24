#!/bin/bash

. ./cmd.sh
. ./path.sh

#Config:
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: local/room_simulator.sh <UID> <conf-dir> <rever_noise> <noise-dir> <wav-in-dir>"
   echo "main options"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

uid=$1
cdir=$2
rn=$3
ndir=$4
adir=$5

set -e
set -o pipefail

person=`echo $uid | awk -F"_" '{print $1}'`
session=`echo $uid | awk -F"_" '{print $2}'`
mic=`echo $uid | awk -F"_" '{print $3}'`
odir=aug/$session/$person
expdir=exp/aug/$session/$person

check_file(){
  if [ ! -f $odir/$mic.finish ]; then
    sleep 5m
    check_file
  fi
}

mkdir -p $odir
mkdir -p $expdir

if [ -f $expdir/${mic}_aug.info ]; then
  check_file
  echo $odir/${session}_${person}_${mic}.txt
else
  local/reverberant_speech.py $adir/${session}_${person}.wav $cdir/${session}.json $rn $ndir $odir $mic > $expdir/${mic}_aug.info
  wait
  touch $odir/$mic.finish
  echo $odir/${session}_${person}_${mic}.txt
fi
