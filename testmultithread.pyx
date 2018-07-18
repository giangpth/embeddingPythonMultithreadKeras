class Tester:
    def __init__(self):
        self.n = 0
    def incre2(self, num, id):
        print('count to {}'.format(num))
        while self.n < num:
            if self.n%100000 == 0:
                print ('n of thread {} at this time self.n is {}'.format(id, self.n))
            self.n = self.n + 1      
        print('now self.n of thread{} is {}'.format(id, self.n))
            

cdef public createInstance():
    print('create')
    return Tester()

cdef public testprocess(object p, id):
    print('run cdef')
    p.incre2(100000000, id)

