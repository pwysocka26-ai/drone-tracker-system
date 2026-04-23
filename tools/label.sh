#!/usr/bin/env bash
# Wrapper around tools/_label_track_init.py with v2 dataset paths baked in.
#
# Short form resolves <clip_id> (e.g. "0002") to data/ext_rgb_drone/DJI_*_<clip_id>_V.MP4
# and uses tag "dji<clip_id>".
# Long form takes explicit video path + tag.
#
# usage:
#   bash tools/label.sh <clip_id> [start_frame=0] [step=10]
#   bash tools/label.sh <video_path> <tag> [start_frame=0] [step=10]

set -e

if [ $# -lt 1 ]; then
    cat <<EOF
usage:
  short: bash tools/label.sh <clip_id> [start_frame=0] [step=10]
  long:  bash tools/label.sh <video_path> <tag> [start_frame=0] [step=10]

short-form examples (auto-resolves data/ext_rgb_drone/DJI_*_<clip_id>_V.MP4):
  bash tools/label.sh 0002
  bash tools/label.sh 0002 1400
  bash tools/label.sh 0002 1400 5
EOF
    exit 1
fi

# Detect form: if first arg looks like a path (contains / or .mp4/.MP4/.avi), use long form.
if [[ "$1" == */* ]] || [[ "$1" == *.MP4 ]] || [[ "$1" == *.mp4 ]] || [[ "$1" == *.avi ]]; then
    if [ $# -lt 2 ]; then
        echo "long form requires <video_path> <tag>" >&2
        exit 1
    fi
    VIDEO="$1"
    TAG="$2"
    START_FRAME="${3:-0}"
    STEP="${4:-10}"
    END_FRAME="${5:-0}"
else
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
    START_FRAME="${2:-0}"
    STEP="${3:-10}"
    END_FRAME="${4:-0}"
fi

echo "video: $VIDEO"
echo "tag:   $TAG  start_frame=$START_FRAME  step=$STEP  end_frame=$END_FRAME"

python tools/_label_track_init.py \
    "$VIDEO" "$TAG" \
    training/v2/images/train \
    training/v2/labels/train \
    training/v2/review \
    "$STEP" "$START_FRAME" "$END_FRAME"
