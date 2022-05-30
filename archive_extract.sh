while read l
do
basename="$(basename ${l%.tar})"
tar tf  &> /dev/null
exitcode=$?
if [ $exitcode -ne 0 ]
then
    echo $basename >> archive_extract_fatal.txt
fi
done < archive2.txt