import socket
import threading
import logging
import datetime

FORMAT="%(asctime)s %(threadtime)s %(thread)d %(message)s"
logging.basicConfig(format=FORMAT,level=logging.INFO)

#TCP Server
class ChatServer:
    def __init__(self,ip='127.0.0.1',port=9999):
        self.addr=(ip,port)
        self.sock=socket.socket()
        self.clients={}
    
    def start(self):
        self.sock.bind(self.addr)
        self.sock.listen() #the service launched

        threading.Thread(target=self.accept,name='accept').start()
    def accept(self):
        while not self.event.is_set:#one thread
            s,raddr=self.sock.accept()#blocked
            logging.info(raddr)
            logging.info(s)
            self.clients[raddr]=s
            threading.Thread(target=self.recv,name='recv',args=(s,addr)).start()
    def recv(self,sock:socket.socket):
        while True:
            try:
                data=sock.recv(1024) #blocked,bytes
                logging.info(data)
            except Exception as e:
                logging.error(e)
                data=b'quit'
            if data==b'quit':
                self.clients.pop(sock.getpeername())
                sock.close()
                break

            msg="ack{}.{} {}".format(sock.getpeername(),datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),data.decode()).encode()
            for s in self.clients.values():
                s.send(msg)

    def stop(self):
        for s in self.clients.values():
            s.close()
        self.sock.close()
        self.event.set()

cs=ChatServer()
cs.start()

while True:
    cmd=input(">>>")
    if cmd.strip()=='quit':
        cs.stop()
        threading.Event.wait(3)
        break
    logging.info(threading.enumerate())