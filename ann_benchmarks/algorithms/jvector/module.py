import socket
from io import StringIO
import sklearn.preprocessing
import os.path
import time

from ..base.module import BaseANN

class JVector(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self._metric = metric
        self._dimension = dimension
        self._m = method_param['M']
        self._bulk = True;
        self._ef_construction = method_param['efConstruction']
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); 
        print("waiting for jvector...")
        while not os.path.exists("/tmp/jvector.sock"): 
            time.sleep(1)
    
        self._conn.connect("/tmp/jvector.sock")
        print("connected!")

    def fit(self, X):

        metric = ""
        if self._metric == "angular":
            metric = "DOT_PRODUCT"
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        elif self._metric == "euclidean":
            metric = "EUCLIDEAN"
        else:
            raise RuntimeError(f"Unknown metric {self._metric}")

        print("creating index...")
        self._conn.send(bytes(f"CREATE {self._dimension} {metric} {self._m} {self._ef_construction}\n", 'ascii'))
        r = str(self._conn.recv(1024), 'ascii')        
        if r != "OK":
            raise RuntimeError(r)
        
        if self._bulk:
            print(f"bulk indexing {len(X)} vectors...")
            X.tofile("/tmp/data.bin")
            self._conn.send(bytes(f"BULKLOAD /tmp/data.bin\n", 'ascii'))
            r = str(self._conn.recv(1024), 'ascii')
            if r != "OK":
                raise RuntimeError(r)    
        else:
            print(f"indexing {len(X)} vectors...")
            batch_size = 1000
            batch = []
            written = 0
            for i, embedding in enumerate(X):
                batch.append(",".join(map(str, embedding)))
                if len(batch) == batch_size:
                    vec = "] [".join(batch)
                    self._conn.send(bytes(f"WRITE [{vec}]\n",'ascii'))
                    batch = []
                    r = str(self._conn.recv(1024), 'ascii')
                    if r != "OK":
                        raise RuntimeError(r)
                    written += batch_size
                    print(f"written {written}")

            if len(batch) > 0:
                vec = "] [".join(batch)
                self._conn.send(bytes(f"WRITE [{vec}]\n",'ascii'))
                batch = []
                r = str(self._conn.recv(1024), 'ascii')
                if r != "OK":
                    raise RuntimeError(r)

        print("optimizing index...")
        self._conn.send(bytes(f"OPTIMIZE\n",'ascii'))
        r = str(self._conn.recv(1024), 'ascii')
        if r != "OK":
            raise RuntimeError(r)        
        print("done!")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search

    def query(self, v, n):
        if self._metric == "angular":
            v = sklearn.preprocessing.normalize([v], axis=1, norm="l2")[0]
        vec = ",".join(map(str, v))
        self._conn.send(bytes(f"SEARCH {self._ef_search} {n} [{vec}]\n", 'ascii'))
        buf = StringIO()
        while True:
            r = self._conn.recv(1024)
            if r == None:
                break
            rstr = str(r, 'ascii')
            buf.write(rstr)
            if rstr.endswith("\n"):
                break;
        
        result = buf.getvalue()

        if not result.startswith("RESULT"):
            raise RuntimeError(result)

        return [int(id.strip()) for id in result[8:-2].split(',')]


    def get_memory_usage(self):
        if self._conn is None:
            return 0
        self._conn.send(bytes("MEMORY\n", 'ascii'))
        r = str(self._conn.recv(1024), 'ascii')
        if not r.startswith("RESULT"):
            raise RuntimeError(r)
        
        #Should already be in KB
        return int(r.split(' ')[1].strip())
    
    def batch_query(self, X, n):
        if self._metric == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")

        qstr = StringIO()
        qstr.write(f"SEARCH {self._ef_search} {n}")
        for i, query in enumerate(X):
            qstr.write(" [")
            qstr.write(",".join(map(str, query)))
            qstr.write("]")

        qstr.write("\n")
        self._conn.send(bytes(qstr.getvalue(),'ascii'))

        buf = StringIO()
        while True:
            r = self._conn.recv(1024)
            if r == None:
                break
            rstr = str(r, 'ascii')
            buf.write(rstr)
            if rstr.endswith("\n"):
                break;
        
        result = buf.getvalue()

        if not result.startswith("RESULT"):
            raise RuntimeError(result)
        
        res = []
        for row in result[8:-2].split('] ['):
            res.append([int(id.strip()) for id in row.split(',')])
        
        print(f"Batch of {len(res)}")
        self.batch_res = res

    def get_batch_results(self):
        return self.batch_res

    def __str__(self):
        return f"JVector(metric={self._metric}, m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
