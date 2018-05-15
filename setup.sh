mkdir -p data
git clone https://github.com/ryancotterell/derviational-paradigms data/derivational-paradigms

if [ ! -f data/unigram-counts.txt ]; then
  wget https://www.seas.upenn.edu/~ddeutsch/acl2018/unigram-counts.txt -O data/unigram-counts.txt
fi

cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.train.src data/train.src
cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.train.trg data/train.trg
cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.dev.src data/dev.src
cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.dev.trg data/dev.trg
cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.test.src data/test.src
cp data/derivational-paradigms/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.test.trg data/test.trg

python -m morphology.vocab \
  --input-src data/train.src \
  --input-trg data/train.trg \
  --output-file data/vocab.txt

# 90/10 split
sed -n 1,3800p data/train.src > data/train.90-10.90.src
sed -n 1,3800p data/train.trg > data/train.90-10.90.trg
sed -n 3801,4222p data/train.src > data/train.90-10.10.src
sed -n 3801,4222p data/train.trg > data/train.90-10.10.trg

# 85/7/8 split
sed -n 1,3589p data/train.src > data/train.85-7-8.85.src
sed -n 1,3589p data/train.trg > data/train.85-7-8.85.trg
sed -n 3590,3885p data/train.src > data/train.85-7-8.7.src
sed -n 3590,3885p data/train.trg > data/train.85-7-8.7.trg
sed -n 3885,4222p data/train.src > data/train.85-7-8.8.src
sed -n 3885,4222p data/train.trg > data/train.85-7-8.8.trg
