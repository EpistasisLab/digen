!/bin/sh

for s in `ls`; do
#  cat $s/README.md | sed 's|docs/||g' > $s/README1.md
mv $s/README1.md $s/README.md
done