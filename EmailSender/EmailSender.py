from smtplib import *
from tkinter import *
import tkinter.messagebox
import tkinter.filedialog
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


class loginPage(object):
    def __init__(self, master, info='Mail Sending System By Tony'):
        self.master = master
        self.mainlabel = Label(master, text=info, justify=CENTER)
        self.mainlabel.grid(row=0, columnspan=3)

        self.user = Label(master, text='Username：', borderwidth=2)
        self.user.grid(row=1, sticky=W)

        self.pwd = Label(master, text='Password：', borderwidth=2)
        self.pwd.grid(row=2, sticky=W)

        self.server = Label(master, text='Server：', borderwidth=2)
        self.server.grid(row=3, sticky=W)

        self.port = Label(master, text='Port：', borderwidth=2)
        self.port.grid(row=4, sticky=W)

        self.userEntry = Entry(master)
        self.userEntry.grid(row=1, column=1, columnspan=2)
        self.userEntry.focus_set()

        self.pwdEntry = Entry(master, show='*')
        self.pwdEntry.grid(row=2, column=1, columnspan=2)

        self.serverEntry = Entry(master)
        self.serverEntry.grid(row=3, column=1, columnspan=2)

        self.portEntry = Entry(master)
        self.portEntry.grid(row=4, column=1, columnspan=2)

        self.loginButton = Button(master, text='Login', borderwidth=6, command=self.login)
        self.loginButton.grid(row=5, column=1)

        self.clearButton = Button(master, text='Clear', borderwidth=6, command=self.clear)
        self.clearButton.grid(row=5, column=2)

        self.sshloginButton = Button(master, text='Use SSH To Login', borderwidth=2, command=self.sshlogin)
        self.sshloginButton.grid(row=5, column=0)

    def login(self):
        self.username = self.userEntry.get().strip()
        self.passwd = self.pwdEntry.get().strip()
        if len(self.username) == 0 or len(self.passwd) == 0 or '@' not in self.username:
            tkinter.messagebox.showwarning('Warning',
                                           'It seems that you have not enter your username or password or the format is wrong!')
            self.clear()
            self.userEntry.focus_set()
            return

        self.getSmtpHost()
        self.connect()

    def connect(self):
        if not self.serverEntry.get():
            HOST = 'smtp.' + self.smtp + '.com'
        else:
            HOST = self.serverEntry.get().strip()
        try:
            if self.portEntry.get():
                self.mySMTP = SMTP_SSL(HOST, int(self.portEntry.get().strip()))
            else:
                if self.smtp == 'gmail':
                    self.mySMTP = SMTP(HOST, 587)
                    self.mySMTP.starttls()
                elif self.username.split('@')[1] == 'mails.tsinghua.edu.cn':
                    self.mySMTP = SMTP_SSL('mails.tsinghua.edu.cn', 465)
                else:
                    self.mySMTP = SMTP_SSL(HOST)
            self.mySMTP.login(self.username, self.passwd)
        except Exception as e:
            tkinter.messagebox.showerror('Connection Error', '%s' % e)
            return

        self.mySendMail = sendMail(self.master, self.mySMTP, self.username)

    def clear(self):
        self.userEntry.delete(0, END)
        self.pwdEntry.delete(0, END)

    def getSmtpHost(self):
        firstSplit = self.username.split('@')[1]
        self.smtp = firstSplit.split('.')[0]

    def sshlogin(self):
        if not self.serverEntry.get():
            HOST = 'smtp.' + self.smtp + '.com'
        else:
            HOST = self.serverEntry.get().strip()
        try:
            if self.portEntry.get():
                self.mySMTP = SMTP(HOST, int(self.portEntry.get().strip()))
            else:
                if self.smtp == 'gmail':
                    self.mySMTP = SMTP(HOST, 587)
                    self.mySMTP.starttls()
                elif self.username.split('@')[1] == 'mails.tsinghua.edu.cn':
                    self.mySMTP = SMTP_SSL('mails.tsinghua.edu.cn', 993)
                else:
                    self.mySMTP = SMTP(HOST)
            self.mySMTP.login(self.username, self.passwd)
        except Exception as e:
            tkinter.messagebox.showerror('Connection Error', '%s' % e)
            return

        self.mySendMail = sendMail(self.master, self.mySMTP, self.username)


class sendMail(object):
    def __init__(self, master, smtp='', sender='', path='', pythonfileRoute=''):
        self.smtp = smtp
        self.sender = sender
        self.path = path
        self.pythonfileRoute = pythonfileRoute
        self.sendPage = Toplevel(master)

        self.sendToLabel = Label(self.sendPage, text='Send To:')
        self.sendToLabel.grid()
        self.sendToEntry = Entry(self.sendPage)
        self.sendToEntry.grid(row=0, column=1)

        self.subjectLabel = Label(self.sendPage, text='Subject:')
        self.subjectLabel.grid(row=1, column=0)
        self.subjectEntry = Entry(self.sendPage)
        self.subjectEntry.grid(row=1, column=1)

        self.fromToLabel = Label(self.sendPage, text='From:')
        self.fromToLabel.grid(row=2, column=0)

        self.formToAdd = Label(self.sendPage, text=self.sender)
        self.formToAdd.grid(row=2, column=1)

        self.sendFile = Label(self.sendPage, text='Your File Path:')
        self.sendFile.grid(row=3, column=0)
        self.sendFileEntry = Entry(self.sendPage)
        self.sendFileEntry.grid(row=3, column=1)

        self.explorerbutton = Button(self.sendPage, text="Use Explorer", command=self.fileexplorer, borderwidth=6)
        self.explorerbutton.grid(row=3, column=2)

        self.sendText = Text(self.sendPage)
        self.sendText.grid(row=4, column=0, columnspan=3)

        self.sendButton = Button(self.sendPage, text='Send', command=self.sendMail)
        self.sendButton.grid(row=5, column=0)

        self.newButton = Button(self.sendPage, text='New Mail', command=self.newMail)
        self.newButton.grid(row=5, column=2)

    def getMailInfo(self):
        try:
            self.sendToAdd = self.sendToEntry.get().strip()
            self.subjectInfo = self.subjectEntry.get().strip()
            self.sendTextInfo = self.sendText.get(1.0, END)

            self.windowsfileRoute = self.sendFileEntry.get().strip()
            self.pythonfileRoute = self.windowsfileRoute.replace('\\', '/')
            match = re.split('/', self.pythonfileRoute)
            self.filename = match[-1]

        except Exception as e:
            tkinter.messagebox.showerror('Error', "%s" % e)

    def sendMail(self):
        self.getMailInfo()
        body = self.sendTextInfo + "\r\n" + '------from python client'
        msg = MIMEMultipart('alternative')
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        msg.attach(MIMEText('<html><body><h1>' + body + '</h1></body></html>', 'html', 'utf-8'))
        msg['From'] = _format_addr('from python client <%s>' % self.sender)
        msg['To'] = _format_addr('<%s>' % self.sendToAdd)
        msg['Subject'] = Header('%s' % self.subjectInfo, 'utf-8').encode()

        if self.path or self.sendFileEntry.get():
            try:
                part = MIMEApplication(open(self.pythonfileRoute, 'rb').read())
                part.add_header('Content-Disposition', 'attachment', filename=self.filename)
                msg.attach(part)
            except Exception as e:
                tkinter.messagebox.showerror('File Error', '%s' % e)

        try:
            self.smtp.sendmail(self.sender, [self.sendToAdd], msg.as_string())
        except Exception as e:
            tkinter.messagebox.showerror('Sending Failed', "%s" % e)
            return
        tkinter.messagebox.showinfo('Congratulations', 'Sending Successfully.')

    def newMail(self):
        self.sendToEntry.delete(0, END)
        self.subjectEntry.delete(0, END)
        self.sendText.delete(1.0, END)

    def fileexplorer(self):
        self.path = tkinter.filedialog.askopenfilename()
        self.sendFileEntry.insert(200, self.path)


if __name__ == '__main__':
    root = Tk()
    root.title('Python Email Client')
    myLogin = loginPage(root)
    mainloop()
