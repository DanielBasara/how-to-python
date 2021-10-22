
from multiprocessing import Process, Queue


def task(num, que):
    num += num
    que.put(num)
    return


if __name__ == "__main__":
    que1 = Queue()
    que2 = Queue()
    t1 = Process(target=task, args=(1, que1,))
    t2 = Process(target=task, args=(2, que2,))
    t1.start()
    t2.start()
    result1 = que1.get()
    result2 = que2.get()
    t1.join()
    t2.join()
    print(result1, result2)
