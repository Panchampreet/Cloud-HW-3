from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.recommendation import ALS
from operator import add
import sys
from docopt import docopt
import contextlib
import itertools

memory = '2g'
name = 'audioScobbler'
master = 'local'

@contextlib.contextmanager
def spark_manager():
    conf = SparkConf().setMaster(master) \
                      .setAppName(name) \
                      .set("spark.executor.memory", memory)
    spark_context = SparkContext(conf=conf)
    try:
        yield spark_context
    finally:
        spark_context.stop()


def parse_plays(line):
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))


def metrics(training_data_file, meta_data):
    tracks = {}
    with open(meta_data, 'r') as open_file:
        tracks = {int(line.split('::')[0]): line.split('::')[1]
                  for line in open_file
                  if len(line.split('::')) == 3}

    with spark_manager() as context:
        plays = context.textFile(training_data_file) \
                         .filter(lambda x: x and len(x.split('::')) == 4) \
                         .map(parse_plays)

        most_played = plays.values() \
                            .map(lambda r: (r[1], 1)) \
                            .reduceByKey(add) \
                            .map(lambda r: (r[1], r[0])) \
                            .sortByKey(ascending=False) \
                            .collect()[:10]


def train(training_data_file, numPartitions, ranks, lambdas, iterations):
    with spark_manager() as context:
        plays = context.textFile(training_data_file) \
                         .filter(lambda x: x and len(x.split('::')) == 4) \
                         .map(parse_play)

        numPlays = plays.count()

        numUsers = plays.values() \
                          .map(lambda r: r[0]) \
                          .distinct() \
                          .count()

        numTracks = plays.values() \
                           .map(lambda r: r[1]) \
                           .distinct() \
                           .count()

        training = plays.filter(lambda x: x[0] < 6) \
                          .values() \
                          .repartition(numPartitions) \
                          .cache()

        validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
                            .values() \
                            .repartition(numPartitions) \
                            .cache()

        test = ratings.filter(lambda x: x[0] >= 8) \
                      .values() \
                      .cache()

        numTraining = training.count()
        numValidation = validation.count()
        numTest = test.count()

        bestModel, bestRank, bestLambda, bestNumIter = None, 0, -1.0, -1
        
        for rank, lmbda, numIter in itertools.product(ranks,
                                                      lambdas,
                                                      iterations):
            model = ALS.train(plays=training,
                              rank=rank,
                              iterations=numIter,
                              lambda_=lmbda)


def recommend(training_data_file, meta_data, user_plays,
              numPartitions, rank, iterations, _lambda):

    my_plays = ((0, 2858, user_plays[0]), (0, 480,  user_plays[1]), (0, 589,  user_plays[2]), (0, 2571, user_plays[3]),
        (0, 1270, user_plays[4]))
    tracks_listened = set([_play[1] for _play in my_plays])

    with spark_manager() as context:
        training = context.textFile(training_data_file) \
                          .filter(lambda x: x and len(x.split('::')) == 4) \
                          .map(parse_play) \
                          .values() \
                          .repartition(numPartitions) \
                          .cache()

        model = ALS.train(training, rank, iterations, _lambda)

        songs_rdd = context.textFile(training_data_file) \
                           .filter(lambda x: x and len(x.split('::')) == 4) \
                           .map(parse_play)

        songs = songs_rdd.values() \
                         .map(lambda r: (r[1], 1)) \
                         .reduceByKey(add) \
                         .map(lambda r: r[0]) \
                         .filter(lambda r: r not in films_seen) \
                         .collect()

        candidates = context.parallelize(tracks) \
                            .map(lambda x: (x, 1)) \
                            .repartition(numPartitions) \
                            .cache()

        predictions = model.predictAll(candidates).collect()

        # getting the top 10 recommendations
        recommendations = sorted(predictions,
                                 key=lambda x: x[2],
                                 reverse=True)[:10]

    tracks = {}

    with open(meta_data, 'r') as open_file:
        tracks = {int(line.split('::')[0]): line.split('::')[1]
                  for line in open_file
                  if len(line.split('::')) == 3}

    for track_id, _, _ in recommendations:
        print tracks[track_id] if track_id in tracks else track_id


def main(argv):
    opt = docopt(__doc__, argv)

    if opt['train']:
        ranks    = [int(rank)      for rank in opt['--ranks'].split(',')]
        lambdas  = [float(_lambda) for _lambda in opt['--lambdas'].split(',')]
        iterations = [int(_iter)   for _iter in opt['--iterations'].split(',')]

        train(opt['<training_data_file>'],
              int(opt['--partitions']),
              ranks,
              lambdas,
              iterations)

    if opt['metrics']:
        metrics(opt['<training_data_file>'],
                opt['<meta_data>'])

    if opt['recommend']:
        ratings = [float(_rating) for _rating in opt['--ratings'].split(',')]
        recommend(training_data_file=opt['<training_data_file>'],
                  meta_data=opt['<meta_data>'],
                  user_ratings=ratings,
                  numPartitions=int(opt['--partitions']),
                  rank=int(opt['--rank']),
                  iterations=int(opt['--iteration']),
                  _lambda=float(opt['--lambda']))


if __name__ == "__main__":
        main(sys.argv[1:])
