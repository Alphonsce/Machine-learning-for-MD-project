N=$3

ns=$1000

outf=$ns

thermof=$100

dt=$0.0005

# ./md -N ${N} -ns ${ns} -outf ${outf} -thermof ${thermof} -dt ${dt}

./md -N 3 -ns 10000 -outf 1 -thermof 100 -dt 0.0005