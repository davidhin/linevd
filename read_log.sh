for f in hpc/logs/code_gnn_evalall_7081*.info
do
echo newffffffile
grep -A9 -e "Loaded model weights from checkpoint" $f \
    | grep -v -e "─" -e "Test metric" -e "^$" \
    | sed -E -e 's/^\s+//g' -e 's/\s+/ /g' -e 's@Loaded model weights from checkpoint at logs_5_truncated_untruncated/@@g'
echo endffffffile
done