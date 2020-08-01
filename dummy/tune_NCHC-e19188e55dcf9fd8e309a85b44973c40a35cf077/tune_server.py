from multiprocessing.connection import Listener
from time import sleep
from datetime import timedelta, datetime
import pprint

pp = pprint.PrettyPrinter(indent=2)

def test_obf(hparam):
    x = hparam["x"]
    y = hparam["y"]
    sleep(10)
    return x**2 + abs(y) + y**3

def server_setup(address, authkey, obf):
    with Listener(address, authkey=authkey) as listener:
        print("server setup!")
        counter=0
        while True:
            with listener.accept() as conn:
                counter+=1
                print("start #{}".format(counter))
                msg = conn.recv()
                if msg == "kill connection!":
                    return
                else:
                    print("hyperparameter:")
                    pp.pprint(msg)
                    start = datetime.now()
                    obf_value = obf(msg)
                    end = datetime.now()
                    elapsed = end - start
                    conn.send("res")
                    conn.send(obf_value)
                    conn.send("info")
                    info = "end #{}, elapsed: {}\n".format(counter, str(elapsed))
                    conn.send(info)

if __name__ == "__main__":
    domain_socket = "/tmp/tuneconn"
    conn_authkey = b'physionet'
    server_setup(domain_socket, conn_authkey, test_obf)