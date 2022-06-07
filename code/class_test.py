


class Test(object):
    def __getitem__(self, index):
        print(index)
        print(type(index))
        return index


if __name__ == '__main__':
    var = Test()
    a = slice(1, 2)
    print(var[1])
    print(var[1:2])
    print(var[a])
    print(var[[1, 2, 3, 4]])

