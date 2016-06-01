from multiprocessing import Pool

class container:
    def __init__(self):
        self.blur = True

    def square(self, a):
        if self.blur:
            a += 5
        return a**2

    def testpool(self):
        blaat = range(100)
        p = Pool()
        print(p.map(self.square, blaat))

a = container()
a.testpool()
