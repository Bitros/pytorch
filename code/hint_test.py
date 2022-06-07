
class BroadcastingListCls(object):
    def __getitem__(self, types):
        return


for i in range(1, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingListCls()


def a(x: BroadcastingList1[int]):
    print(x)


def b(x: BroadcastingList2[int]):
    print(x)


if __name__ == '__main__':
    # just show type hint in parameter level
    a(0)
    b(1)
