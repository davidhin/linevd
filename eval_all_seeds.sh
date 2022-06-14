for i in 0 1 2 3 4
do
sbatch --job-name cgdt$i hpc/code_gnn_evalall.sh _ABS_DATAFLOW_datatype_all sum $i
sbatch --job-name cgall$i hpc/code_gnn_evalall.sh _ABS_DATAFLOW_api_datatype_literal_operator_all sum $i

sbatch --job-name cgdtiu$i hpc/code_gnn_evalall.sh _ABS_DATAFLOW_datatype_all_includeunknown sum $i
sbatch --job-name cgalliu$i hpc/code_gnn_evalall.sh _ABS_DATAFLOW_api_datatype_literal_operator_all_includeunknown sum $i
done
