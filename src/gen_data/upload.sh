FILES=*.xml
for f in $FILES
	do
	echo "Uploading $f file ..."
	davix-put --userlogin $DCUNAME --userpass $DCPASS $f davs://io.erda.dk/DeepContact/2DSim/$f
	ret=$?
	echo "return code $ret"
	if [ $ret -ne 0 ]; then
		echo "retrying ..."
		davix-put --userlogin $DCUNAME --userpass $DCPASS $f davs://io.erda.dk/DeepContact/2DSim/$f
	fi
done