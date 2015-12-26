__author__ = 'sagabhan, ctewani'


class Cache:
    train = []

    test = []
    trainFeatures = []
    testFeatures = []

    @staticmethod
    def read_data(file_name, is_test):
        exemplars = []
        feature_vectors = []
        file = open(file_name, 'r');
        for line in file:
            line = line.split()
            feature_vector = []  # int(x) for x in line[2:]
            for x in line[2:]:
                x = int(x)
                feature_vector.append(x)

            exemplars += [(line[0], line[1], feature_vector)]
            feature_vectors.append(feature_vectors)

        if is_test:
            Cache.test = exemplars
            Cache.testFeatures = feature_vectors
        else:
            Cache.train = exemplars
            Cache.trainFeatures = feature_vectors
