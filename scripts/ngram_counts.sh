# Downloads the Google Ngram counts for all unigrams and aggregates the
# counts over the year counts.
#
# Arguments:
#   1: The directory where the ngram counts will be downloaded
#   2: The file where the aggreated counts will be saved

DOWNLOAD_DIR=$1
OUTPUT_FILE=$2

mkdir -p ${DOWNLOAD_DIR}
mkdir -p $(dirname ${OUTPUT_FILE})

for LETTER in 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x' 'y' 'z'
do
  FILE="googlebooks-eng-all-1gram-20120701-$LETTER.gz"
  if [ ! -f $DOWNLOAD_DIR/$FILE ]; then
    wget http://storage.googleapis.com/books/ngrams/books/$FILE -O $DOWNLOAD_DIR/$FILE
  fi
done

python scripts/ngram_counts.py $DOWNLOAD_DIR $OUTPUT_FILE
