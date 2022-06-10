set -x

feat="_ABS_DATAFLOW_api_datatype_literal_operator"
dirname="bigvul_linevd_codebert_cfg_$feat"
tarname="$dirname.tar.gz"
ssh benjis@prontodtn.las.iastate.edu "cd /work/LAS/weile-lab/benjis/weile-lab/linevd/storage/cache; time tar zcf $tarname $dirname"
time rsync -a --info=progress2 "benjis@prontodtn.las.iastate.edu:/work/LAS/weile-lab/benjis/weile-lab/linevd/storage/cache/$tarname" "storage/cache/"
(cd storage/cache/; time tar zxf $tarname)