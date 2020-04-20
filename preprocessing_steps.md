Steps to follow for Recomb data preprocessing
---

###Run in command line

For creating the genomic "index" for the double stranded breaks
```
awk -F '\t' '!/^#/' DSB_hotspots.txt | awk -F '\t' '{print $1":"$2"-"$3"(+)"}' DSB_hotspots.txt | tail -n +28 > DSB_hotspots_ID.txt
```
copy the **encode_roadmap_act.txt** from your Basset directory to this current one; then we want to merge the two datasets
###Sanity check
Make sure that there are no overlaps in genomic regions
```
awk -F '\t' '{print $1}' encode_roadmap_act.txt > encode_roadmap_act_col1.txt

sort DSB_hotspots_ID.txt encode_roadmap_act_col1.txt | uniq -d > duplicate.txt

```
### No overlaps! Let's merge!
```
sort DSB_hotspots_ID.txt encode_roadmap_act_col1.txt | uniq -u > DSB_encode_roadmap.txt
```

###Sanity check:
To count the number of genomic regions: 
```
wc -l DSB_encode_roadmap.txt # 2083997
wc -l encode_roadmap_act.txt # 2021887
```
the difference is 62110, which is
```
wc -l DSB_hotspots.txt
```
**_yay!_**

_The DSB_encode_roadmap.txt file has all of the regions we care about_.

Now we need to identify whether they are from DSB (1) or not (0). Run R script
```
Rscript make_act.R
```

Then create bed file
```
sed 's/:/\t/g; s/-/\t/g; s/(+)/\t/g' activity.txt | awk -F '\t' '{print $1, $2, $3, $6}' | tail -n +2 |  sed 's/$/ ./g; s/$/ ./g; s/$/ +/g' |  awk -F ' ' '{print $1, $2, $3, $5, $6, $7, $4}' | sed 's/ /\t/g' > DSB_encode_roadmap.bed
```

### Sanity check:
Make sure that you have the correct # of 1s and 0s
```
awk '{if ($7 ~ "0") {print}}' DSB_encode_roadmap.bed | lesss | wc -l
```
should be : 62110
```
awk '{if ($7 ~ "1") {print}}' DSB_encode_roadmap.bed | lesss | wc -l
```

Merge all FASTA files into a single one --> hg19 reference format. 
(Must cd to the /fasta directory)
```
cat *.fa > hg19.fa
```

Convert sequences to FASTA (needed for Torch)
```
bedtools getfasta -fi ../data/fasta/hg19.fa -bed DSB_encode_roadmap.bed -s -fo DSB_encode_roadmap.fa
```

#TO-DO: de-bug fasta_to_hdf5.py file
convert fasta to hdf5 using this command
```
./fasta_to_hdf5.py -c -r -t 71886 -v 70000 ./fasta/hg19.fa activity.txt DSB_encode_roadmap.h5
```
If script is not running because lack of permission, run this code:
```
chmod u+x <filename.py>
```
