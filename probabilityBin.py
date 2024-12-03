class ProbBucket:
    def __init__(self,probability = 0.0, limit = (0, 0)):
        self.probability = probability
        self.limit = limit

    def checkInlimit(self, value):
        lowerlimit, upperlimit = self.limit
        if lowerlimit == -1:
            if value <= upperlimit:
                return True
        if upperlimit == -1 :
            if value > lowerlimit: 
                return True
        if value > lowerlimit and value <= upperlimit:
            return True
        return False

def getMinMax(values):
    min = values[0]
    max = values[0]
    for i in range(len(values)):
        if min > values[i]:
            min = values[i]

        if max < values[i]:
            max = values[i]
    return min, max    

class AttributeProbabilityBin:
    def __init__(self, values, noofbuckets, limits = (0, 0)):
        self.values = values
        min, max = limits
        if min == 0 and max == 0:
            min, max = getMinMax(values)
        valuerange = max - min
        bucketsize = valuerange / noofbuckets
        bucket = ProbBucket()
        self.probalityBuckets = [bucket] * (noofbuckets+2)
        self.probalityBuckets[0] = ProbBucket(probability=0, limit=(-1, min))
        for i in range(noofbuckets):
            self.probalityBuckets[i+1] = ProbBucket(probability=0, limit=(min+(i*bucketsize), min+((i+1)*bucketsize)))
        self.probalityBuckets[noofbuckets+1] = ProbBucket(probability=0, limit=(max, -1))
        
        
        for i in range(len(values)):
            for j in range(len(self.probalityBuckets)):
                bucket = self.probalityBuckets[j]
                if bucket.checkInlimit(values[i]):
                    bucket.probability = bucket.probability + 1
                    self.probalityBuckets[j] = bucket
                    break
        
        for i in range(len(self.probalityBuckets)):
            bucket = self.probalityBuckets[i]
            bucket.probability = bucket.probability/len(values)
            self.probalityBuckets[i] = bucket
    
    def getProbability(self, value):
        for i in range(len(self.probalityBuckets)):
            bucket = self.probalityBuckets[i]
            if bucket.checkInlimit(value):
                return bucket.probability