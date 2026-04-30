#!/usr/bin/env bash
# Wrapper around tools/_label_track_init.py — multi-class CSRT labeller dla v6.
#
# Multi-class flow: ten sam klip labellowany 2-3x (raz per klasa). Tag jest
# WSPÓLNY dla wszystkich klas na tym klipie -> JPG zapisany raz, label
# .txt ma append mode -> kazda klasa dodaje swoja linie do tego samego pliku.
#
# Klasy: 0=dron_maly, 1=dron_duzy, 2=pilka. Akceptujemy nazwy (maly/duzy/pilka)
# albo numery.
#
# usage:
#   short (CSTN-XXXX z desktop Chrcynno):
#     bash tools/label.sh CSTN-XXXX <class> [start_frame=0] [step=10] [end_frame=0]
#   long (explicit video + tag):
#     bash tools/label.sh <video_path> <tag> <class> [start_frame=0] [step=10] [end_frame=0]
#   archive DJI (back-compat z v2/v3):
#     bash tools/label.sh <clip_id> <class> [start_frame=0] [step=10] [end_frame=0]

set -e

CHRCYNNO_DIR="/c/Users/Paulina Wysocka/Desktop/30.04.2026-Chrcynno"

class_to_id() {
    case "$1" in
        0|maly|dron_maly) echo 0 ;;
        1|duzy|dron_duzy) echo 1 ;;
        2|pilka) echo 2 ;;
        *) echo "unknown class: $1 (use maly|duzy|pilka or 0|1|2)" >&2; exit 1 ;;
    esac
}

if [ $# -lt 2 ]; then
    cat <<EOF
usage:
  CSTN:    bash tools/label.sh CSTN-XXXX <class> [start_frame=0] [step=10] [end_frame=0]
  long:    bash tools/label.sh <video_path> <tag> <class> [start_frame=0] [step=10] [end_frame=0]
  DJI:     bash tools/label.sh <clip_id> <class> [start_frame=0] [step=10] [end_frame=0]

class: maly | duzy | pilka  (albo 0 | 1 | 2)

examples:
  bash tools/label.sh CSTN-1353 duzy 5750 10 6750
  bash tools/label.sh CSTN-1353 pilka 5750 10 6750
  bash tools/label.sh CSTN-1254 maly 0 20
EOF
    exit 1
fi

# Detect form
if [[ "$1" == CSTN-* ]]; then
    # short CSTN form
    CLIP_ID="${1#CSTN-}"
    VIDEO="$CHRCYNNO_DIR/$1.mkv"
    if [ ! -f "$VIDEO" ]; then
        echo "cannot find $VIDEO" >&2; exit 1
    fi
    TAG="chrcynno${CLIP_ID}"
    CLASS_ID=$(class_to_id "$2")
    START_FRAME="${3:-0}"
    STEP="${4:-10}"
    END_FRAME="${5:-0}"
elif [[ "$1" == */* ]] || [[ "$1" == *.MP4 ]] || [[ "$1" == *.mp4 ]] || [[ "$1" == *.avi ]] || [[ "$1" == *.mkv ]]; then
    # long form: explicit path + tag + class
    if [ $# -lt 3 ]; then
        echo "long form requires <video_path> <tag> <class>" >&2
        exit 1
    fi
    VIDEO="$1"
    TAG="$2"
    CLASS_ID=$(class_to_id "$3")
    START_FRAME="${4:-0}"
    STEP="${5:-10}"
    END_FRAME="${6:-0}"
else
    # short DJI form (back-compat)
    CLIP_ID="$1"
    # shellcheck disable=SC2206
    MATCHES=( data/ext_rgb_drone/DJI_*_${CLIP_ID}_V.MP4 )
    if [ ${#MATCHES[@]} -ne 1 ] || [ ! -f "${MATCHES[0]}" ]; then
        echo "cannot resolve clip_id '$CLIP_ID' to a unique file in data/ext_rgb_drone/" >&2
        echo "matches: ${MATCHES[*]}" >&2
        exit 1
    fi
    VIDEO="${MATCHES[0]}"
    TAG="dji${CLIP_ID}"
    CLASS_ID=$(class_to_id "$2")
    START_FRAME="${3:-0}"
    STEP="${4:-10}"
    END_FRAME="${5:-0}"
fi

CLASS_NAME=$(awk -v id="$CLASS_ID" 'BEGIN{split("dron_maly,dron_duzy,pilka",a,","); print a[id+1]}')

echo "video:    $VIDEO"
echo "tag:      $TAG"
echo "class:    $CLASS_NAME (id=$CLASS_ID)"
echo "frames:   start=$START_FRAME  step=$STEP  end=$END_FRAME"

python tools/_label_track_init.py \
    "$VIDEO" "$TAG" \
    training/v6/images/train \
    training/v6/labels/train \
    training/v6/review \
    "$STEP" "$START_FRAME" "$END_FRAME" "$CLASS_ID"
