import numpy as np


class Heap:

    def __init__(self):
        self.h = []
        self.mapping = dict()

    def __iter__(self):
        self.entries = sorted([x for x in self.h], key=lambda x : -x[0])
        self.index = 0
        return self

    def __next__(self):
        try:
            result = self.entries[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    def push(self, r, v):
        # push on heap
        entry = [r, v]
        self.h.append(entry)
        self.mapping[v] = len(self.h) - 1
        self.sift_up(len(self.h) - 1)

    def get_entry(self, v):
        try:
            return self.h[self.mapping[v]]
        except KeyError:
            return None
        # except IndexError:
        #     return None

    def sift_up(self, idx):
        parent = int(idx / 2)
        while idx > 0 and self.h[idx][0] > self.h[parent][0]:
            # exchange
            tmp = self.h[parent]
            self.h[parent] = self.h[idx]
            self.h[idx] = tmp
            # swap entries in mapping
            self.mapping[self.h[parent][1]] = parent
            self.mapping[self.h[idx][1]] = idx
            # move pointer one level up
            idx = parent
            parent = int(idx / 2)

    def pop(self):
        if len(self.h) == 0:
            return None
        else:
            # remove root entry and put right-most item into root
            res = self.h[0]
            self.h[0] = self.h[-1]
            self.h = self.h[:-1]
            # update mapping
            del self.mapping[res[1]]
            if len(self.h) > 0:
                self.mapping[self.h[0][1]] = 0
                self.sift_down(0)
            return res

    def sift_down(self, idx):
        while int(2 * idx + 1) < len(self.h):
            child = int(2 * idx + 1)

            if child < len(self.h) - 1 and self.h[child][0] < self.h[child+1][0]:
                child += 1

            if self.h[idx][0] > self.h[child][0]:
                break
            # exchange
            tmp = self.h[child]
            self.h[child] = self.h[idx]
            self.h[idx] = tmp
            # swap entries in mapping
            self.mapping[self.h[child][1]] = child
            self.mapping[self.h[idx][1]] = idx

            idx = child

    def update(self, r, v):
        if v not in self.mapping:
            return False
        idx = self.mapping[v]
        old_r = self.h[idx][0]
        self.h[idx][0] = r
        if old_r < r:
            self.sift_up(idx)
        elif old_r > r:
            self.sift_down(idx)
        return True

    def remove(self, v):
        if v not in self.mapping:
            return False
        # update the heap
        idx = self.mapping[v]
        self.h[idx] = self.h[-1]
        self.h = self.h[:-1]
        # update the mapping
        if idx < len(self.h):
            self.mapping[self.h[idx][1]] = idx
        else:
            assert idx == len(self.h)
        del self.mapping[v]
        self.sift_down(idx)
        return True

    def is_empty(self):
        return len(self.h) == 0

    def print_heapsorted(self, destruct=False):
        if destruct:
            while not self.is_empty():
                print(self.pop())
        else:
            copy = Heap()
            for e in self.h:
                copy.push(e[0], e[1])
            while not copy.is_empty():
                print(copy.pop())


if __name__ == '__main__':
    h = Heap()
    np.random.seed(123)
    vals = [(r, idx) for idx, r in enumerate(np.random.rand(20))]
    for v in vals:
        h.push(*v)

    h.print_heapsorted()

    s = h.update(1.273757236535, 12)
    print('Update successful:', s)

    h.print_heapsorted()

    s = h.remove(8)
    print('Remove successful:', s)

    h.print_heapsorted()

    s = h.update(0.6666666666666, 19)
    print('Update successful:', s)

    h.print_heapsorted()

    toremove = np.arange(20)
    np.random.shuffle(toremove)
    toremove = list(toremove)
    for v in toremove:
        s = h.remove(v)
        print('Removal of {x} successful: {r}'.format(x=v, r=s))

    print('check whether everything is empty')
    print(len(h.mapping))
    print(len(h.h))
