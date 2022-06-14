# for i in 2 3 4  # NOTE: remaining seeds after partial run
for i in 0 1 2 3 4
do
sbatch --wait --job-name cgdt$i hpc/code_gnn_evaldataset.sh _ABS_DATAFLOW_datatype_all $i
sbatch --wait --job-name cgall$i hpc/code_gnn_evaldataset.sh _ABS_DATAFLOW_api_datatype_literal_operator_all $i

sbatch --wait --job-name cgdtiu$i hpc/code_gnn_evaldataset.sh _ABS_DATAFLOW_datatype_all_includeunknown $i
sbatch --wait --job-name cgalliu$i hpc/code_gnn_evaldataset.sh _ABS_DATAFLOW_api_datatype_literal_operator_all_includeunknown $i
done
