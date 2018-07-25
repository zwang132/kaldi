#!/bin/bash
#
# Begin configuration section
nj=96
decode_nj=20
stage=0
enhancement=beamformit
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e # exit on error
chime5_corpus=/export/corpora4/CHiME5
main_dir=/export/b05/zhiqiw
json_dir=${chime5_corpus}/transcriptions
method=$1
dset=$2

if [ $stage -le 1 ]; then
  # Beamforming using reference arrays
  # enhanced WAV directory
  enhandir=enhan_${method}
  for mictype in u01 u02 u03 u04 u05 u06; do
    local/run_beamformit.sh --cmd "$train_cmd" \
	                    ${main_dir}/${dset}_${method} \
			    ${enhandir}/${dset}_${enhancement}_${method}_${mictype} \
                            ${mictype}
  done

  local/prepare_data.sh --mictype ref "$PWD/${enhandir}/${dset}_${enhancement}_${method}_u0*" \
	                ${json_dir}/${dset} data/${dset}_${enhancement}_${method}_ref
fi

if [ $stage -le 2 ]; then
  for set_name in ${dset}_${enhancement}_${method}_ref; do
    utils/copy_data_dir.sh data/${set_name} data/${set_name}_nosplit
    mkdir -p data/${set_name}_nosplit_fix
    cp data/${set_name}_nosplit/{segments,text,wav.scp} data/${set_name}_nosplit_fix/
    awk -F "_" '{print $0 "_" $3}' data/${set_name}_nosplit/utt2spk > data/${set_name}_nosplit_fix/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${set_name}_nosplit_fix/utt2spk > data/${set_name}_nosplit_fix/spk2utt
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${set_name}_nosplit_fix data/${set_name}
  done
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc
  for x in ${dset}_${enhancement}_${method}_ref; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
	               data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi

