#!/bin/bash

. ./cmd.sh
. ./path.sh


# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=sdm1

. utils/parse_options.sh

for d in exp/${mic}/tri*/decode_*pr1-7; do
  grep Sum $d/*scor*/*ys | utils/best_wer.sh
done
for d in exp/${mic}/chain*/tdnn*/decode_*; do
  grep Sum $d/*scor*/*ys | utils/best_wer.sh
done

exit 0
