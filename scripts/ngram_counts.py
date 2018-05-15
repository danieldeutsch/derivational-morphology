import gzip
import sys
from collections import defaultdict
from glob import glob
from tqdm import tqdm


def main():
    """Calculates the frequency of each unigram using the Google Ngram data.

    Arguments:
        data_dir: The directory that contains the Google Ngram gz files.
        output_file: The file to save the ngram counts.
    """
    data_dir = sys.argv[1]
    output_file = sys.argv[2]

    with open(output_file, 'w') as out:
        for path in tqdm(sorted(glob(data_dir + '/*.gz'))):
            counts = defaultdict(int)
            with gzip.open(path, 'r') as f:
                for line in tqdm(f):
                    # Each line is ngram, year, total count, volume count
                    ngram, _, count, _ = line.decode().strip().split('\t')

                    # Maybe remove the POS tag if it exists. Here, a POS tag
                    # exists if all of the letters after the last underscore
                    # are capitalized.
                    index = ngram.rfind('_')
                    if index >= 0 and ngram[index + 1:].isupper():
                        ngram = ngram[:index]

                    counts[ngram] += int(count)

            for ngram, count in sorted(counts.items()):
                out.write('{}\t{}\n'.format(ngram, str(count)))


if __name__ == '__main__':
    main()
