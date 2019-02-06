#!/bin/bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
# Apache 2.0

# Begin configuration section.
mictype=worn # worn, ref or others
datatype=
cleanup=true
rn=0
cdir=${adir}/../../conf
ndir=${adir}/../../noise/train
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <json-transcript-dir> <output-dir>"
  echo -e >&2 "eg:\n  $0 /corpora/chime5/audio/train /corpora/chime5/transcriptions/train data/train"
  exit 1
fi

set -e -o pipefail

adir=$1
jdir=$2
dir=$3

#json_count=$(find $jdir -name "*.json" | wc -l)
#wav_count=$(find $adir -name "*.wav" | wc -l)
json_count=$(find $jdir -name "*.json" | wc -l | awk '{print $1}')
wav_count=$(find $adir -name "*.wav" | wc -l | awk '{print $1}')

if [ "$json_count" -eq 0 ]; then
  echo >&2 "We expect that the directory $jdir will contain json files."
  echo >&2 "That implies you have supplied a wrong path to the data."
  exit 1
fi
if [ "$wav_count" -eq 0 ]; then
  echo >&2 "We expect that the directory $adir will contain wav files."
  echo >&2 "That implies you have supplied a wrong path to the data."
  exit 1
fi

echo "$0: Converting transcription to text"

mkdir -p $dir
for file in $jdir/*json; do
  ./local/json2text.py --mictype $mictype $file
done | \
  sed -e "s/\[inaudible[- 0-9]*\]/[inaudible]/g" |\
  sed -e 's/ - / /g' |\
  sed -e 's/mm-/mm/g' > $dir/text.orig

echo "$0: Creating datadir $dir for type=\"$mictype\""

if [ $mictype == "worn" ]; then
  # convert the filenames to wav.scp format, use the basename of the file
  # as a the wav.scp key, add .L and .R for left and right channel
  # i.e. each file will have two entries (left and right channel)
  find $adir -name  "S[0-9]*_P[0-9]*.wav" | \
    perl -ne '{
      chomp;
      $path = $_;
      next unless $path;
      @F = split "/", $path;
      ($f = $F[@F-1]) =~ s/.wav//;
      @F = split "_", $f;
      print "${F[1]}_${F[0]}.L sox $path -t wav - remix 1 |\n";
      print "${F[1]}_${F[0]}.R sox $path -t wav - remix 2 |\n";
    }' | sort > $dir/wav.scp

  # generate the transcripts for both left and right channel
  # from the original transcript in the form
  # P09_S03-0006072-0006147 gimme the baker
  # create left and right channel transcript
  # P09_S03.L-0006072-0006147 gimme the baker
  # P09_S03.R-0006072-0006147 gimme the baker
  sed -n 's/  *$//; h; s/-/\.L-/p; g; s/-/\.R-/p' $dir/text.orig | sort > $dir/text
elif [ $mictype == "ref" ]; then
  # fixed reference array

  # first get a text, which will be used to extract reference arrays
  perl -ne 's/-/.ENH-/;print;' $dir/text.orig | sort > $dir/text

  find $adir | grep "\.wav" | sort > $dir/wav.flist
  # following command provide the argument for grep to extract only reference arrays
  grep `cut -f 1 -d"-" $dir/text | awk -F"_" '{print $2 "_" $3}' | sed -e "s/\.ENH//" | sort | uniq | sed -e "s/^/ -e /" | tr "\n" " "` $dir/wav.flist > $dir/wav.flist2
  paste -d" " \
	<(awk -F "/" '{print $NF}' $dir/wav.flist2 | sed -e "s/\.wav/.ENH/") \
	$dir/wav.flist2 | sort > $dir/wav.scp
elif [ $mictype == "aug" ]; then
  if [ $datatype == "Image" ]; then
    # convert filenames to wav.scp format, use the basename of the file
    find $adir -name "S[0-9]*_P[0-9]*.wav" | \
      perl -ne '{
        chomp;
        $path = $_;
        next unless $path;
        @F = split "/", $path;
        ($f = $F[@F-1]) =~ s/.wav//;
        @F = split "_", $f;
        $t = "${F[1]}_${F[0]}";
        @mic_seq = ('U01', 'U02', 'U04', 'U05', 'U06');
        foreach $mic (@mic_seq) {
          print "${t}_${mic}.AUG.L ffmpeg -safe 0 -f concat -i /export/b03/zhiqiw/Reverberant_speech/w_noise/Image_method/${F[0]}/${F[1]}/${F[0]}_${F[1]}_${mic}.txt -map_channel 0.0.0 -f wav pipe: 2>/dev/null |\n";
          print "${t}_${mic}.AUG.R ffmpeg -safe 0 -f concat -i /export/b03/zhiqiw/Reverberant_speech/w_noise/Image_method/${F[0]}/${F[1]}/${F[0]}_${F[1]}_${mic}.txt -map_channel 0.0.1 -f wav pipe: 2>/dev/null |\n";
        }
      }' | sort > $dir/wav.scp.t
  elif [ $datatype == "3D" ]; then
    find $adir -name "S[0-9]*_P[0-9]*.wav" | \
      perl -ne '{
        chomp;
        $path = $_;
        next unless $path;
        @F = split "/", $path;
        ($f = $F[@F-1]) =~ s/.wav//;
        @F = split "_", $f;
        $t = "${F[1]}_${F[0]}";
        @mic_seq = ('U01', 'U02', 'U04', 'U05', 'U06');
        foreach $mic (@mic_seq) {
          print "${t}_${mic}.AUG.L ffmpeg -safe 0 -f concat -i /export/b03/zhiqiw/Reverberant_speech/w_noise/3D_simulation/${F[0]}/${F[1]}/${F[0]}_${F[1]}_${mic}.txt -map_channel 0.0.0 -f wav pipe: 2>/dev/null |\n";
          print "${t}_${mic}.AUG.R ffmpeg -safe 0 -f concat -i /export/b03/zhiqiw/Reverberant_speech/w_noise/3D_simulation/${F[0]}/${F[1]}/${F[0]}_${F[1]}_${mic}.txt -map_channel 0.0.1 -f wav pipe: 2>/dev/null |\n";
        }
      }' | sort > $dir/wav.scp.t
  fi  
  p="\` "
  cat $dir/wav.scp.t | sed "s@$p@\ $cdir\ $rn\ $ndir\ $adir$p@g" | sort > $dir/wav.scp

  # Get a text
  cat $dir/text.orig | \
    perl -ne '{
      chomp;
      $name = $_;
      next unless $name;
      @F = split "_", $name;
      @G = split "-", $F[@F-1];
      @mic_seq = ('U01', 'U02', 'U04', 'U05', 'U06');
      foreach $mic (@mic_seq) {
        print "${F[0]}_${F[1]}_${mic}_${G[0]}.AUG.L-${G[1]}-${G[2]}\n";
        print "${F[0]}_${F[1]}_${mic}_${G[0]}.AUG.R-${G[1]}-${G[2]}\n";
      }
    }' | sort > $dir/text

else
  # array mic case
  # convert the filenames to wav.scp format, use the basename of the file
  # as a the wav.scp key
  find $adir -name "*.wav" -ipath "*${mictype}*" |\
    perl -ne '$p=$_;chomp $_;@F=split "/";$F[$#F]=~s/\.wav//;print "$F[$#F] $p";' |\
    sort -u > $dir/wav.scp

  # convert the transcripts from
  # P09_S03-0006072-0006147 gimme the baker
  # to the per-channel transcripts
  # P09_S03_U01_NOLOCATION.CH1-0006072-0006147 gimme the baker
  # P09_S03_U01_NOLOCATION.CH2-0006072-0006147 gimme the baker
  # P09_S03_U01_NOLOCATION.CH3-0006072-0006147 gimme the baker
  # P09_S03_U01_NOLOCATION.CH4-0006072-0006147 gimme the baker
  perl -ne '$l=$_;
    for($i=1; $i<=4; $i++) {
      ($x=$l)=~ s/-/.CH\Q$i\E-/;
      print $x;}' $dir/text.orig | sort > $dir/text

fi
$cleanup && rm -f $dir/text.* $dir/wav.scp.* $dir/wav.flist

# Prepare 'segments', 'utt2spk', 'spk2utt'
if [ $mictype == "worn" ] || [ $mictype == "aug" ]; then
  cut -d" " -f 1 $dir/text | \
    awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $1, $2/100.0, $3/100.0)}' |\
    sed -e "s/_[A-Z]*\././2" \
    > $dir/segments
elif [ $mictype == "ref" ]; then
  cut -d" " -f 1 $dir/text | \
    awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $1, $2/100.0, $3/100.0)}' |\
    sed -e "s/_[A-Z]*\././2" |\
    sed -e "s/ P.._/ /" > $dir/segments
else
  cut -d" " -f 1 $dir/text | \
    awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $1, $2/100.0, $3/100.0)}' |\
    sed -e "s/_[A-Z]*\././2" |\
    sed -e 's/ P.._/ /' > $dir/segments
fi
cut -f 1 -d ' ' $dir/segments | \
  perl -ne 'chomp;$utt=$_;s/_.*//;print "$utt $_\n";' > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# Check that data dirs are okay!
utils/validate_data_dir.sh --no-feats $dir || exit 1
