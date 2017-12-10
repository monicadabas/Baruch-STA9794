from pyspark import SparkContext

sc = SparkContext("local", "Pi Leibniz approximation")
iteration=10000
partition=4
data = range(0,iteration)
distIn = sc.parallelize(data,partition)
result=distIn.map(lambda n:(1 if n%2==0 else -1)/float(2*n+1)).reduce(lambda a,b: a+b)
print "Pi is %f" % (result*4)
