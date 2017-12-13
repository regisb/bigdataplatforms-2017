from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec

sc = SparkContext()

# Change the IP address to point to your HDFS namenode
lines = sc.textFile("hdfs://127.0.0.1:9000/data/iliad.oneline.txt")
rdd = lines.map(lambda line: line.strip().split())\
    .map(lambda words: [w.strip(",.:;'\"-?!") for w in words])\
    .map(lambda words: [w for w in words if w])\
    .filter(lambda s: s)

# The seed is supposed to produce reproducible results
word2vec = Word2Vec().setSeed(1).setVectorSize(200)
model = word2vec.fit(rdd)
vectors = model.getVectors()

def minus(vec1, vec2):
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]
def plus(vec1, vec2):
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

synonyms = model.findSynonyms(plus(minus(vectors["Priam"], vectors["Hector"]), vectors["Achilles"]), 10)
print(list(synonyms))
