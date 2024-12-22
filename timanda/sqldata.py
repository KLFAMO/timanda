import getpass
import MySQLdb as db
import numpy as np
from . import time_tools as tit
import socket
import os
import pickle as pkl
import struct
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

# Retrieve configuration from .env
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
KEY_FILE = os.getenv("KEY_FILE")
ENCRYPTED_PASSWORD_FILE = os.getenv("ENCRYPTED_PASSWORD_FILE")


def generate_key():
    """Generate and save an encryption key."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)


def load_key():
    """Load the encryption key."""
    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError(f"Key file not found: {KEY_FILE}")
    with open(KEY_FILE, "rb") as key_file:
        return key_file.read()


def save_encrypted_password(password):
    """Encrypt and save the password to a file."""
    key = load_key()
    fernet = Fernet(key)
    encrypted_password = fernet.encrypt(password.encode())
    with open(ENCRYPTED_PASSWORD_FILE, "wb") as file:
        file.write(encrypted_password)


def load_encrypted_password():
    """Load and decrypt the password from the encrypted file."""
    if not os.path.exists(ENCRYPTED_PASSWORD_FILE):
        return None
    key = load_key()
    fernet = Fernet(key)
    with open(ENCRYPTED_PASSWORD_FILE, "rb") as file:
        encrypted_password = file.read()
    return fernet.decrypt(encrypted_password).decode()


def configure():
    """Prompt the user for configuration and save the password securely."""
    if not DB_HOST or not DB_USER or not DB_NAME:
        print("Please ensure DB_HOST, DB_USER, and DB_NAME are set in the .env file.")
        return

    password = load_encrypted_password()
    if not password:
        from getpass import getpass
        password = getpass("Enter database password: ")
        save_encrypted_password(password)
        print("Password saved securely.")

    print("Configuration saved successfully.")


def connect():
    """Connect to the MySQL database using the stored configuration in .env.    
    Returns:
        handler to connection
    """
    password = load_encrypted_password()
    if not password:
        print("Password not found. Run configure() first.")
        return None
        
    # return db.connect( host=host, user=user, password=password, db='measurements')
    try:
        connection = db.connect(
            host=DB_HOST,
            user=DB_USER,
            password=password,
            database=DB_NAME
        )
        if connection:
            print("Connected to MySQL server!")
            return connection
    except Error as e:
        print(f"Error: {e}")
    return None


def gettables():
    """Get all tables in database
    Returns:
        list: list of tables names
    """
    res = dbquery('SHOW tables')
    return [ x[0] for x in res ]


def dbquery(sql_query):
    """ Connect to database and send SQL query
    Returns:
        SQL response or None if error
    """
    try:
        con = connect()
        cur = con.cursor()
        cur.execute(sql_query)
        response = cur.fetchall()
        cur.close()
        con.close()
        return response
    except:
        print('Error: problem with database query sending')
        return None


def dbquery_rm(querstr):
    """Sending SQL query through TCP server
        Allows communication with database from remote computers
        is database is not directly accesable
    """
    s = socket.socket()
    s.connect(('158.75.4.161', 12346))
    ss='dbq:'+querstr
    s.send(ss.encode())
    #receiving
    buffer = 1024
    received = 0
    chunks = []
    while received<4:
        data = s.recv(4-received)
        received += len(data)
        chunks.append(data)
    fsize = struct.unpack('!I',b''.join(chunks))[0]
    #print('Size: ',fsize)

    received = 0
    chunks = []
    while received < fsize:
        data = s.recv(min(fsize-received, buffer)  )
        received += len(data)
        chunks.append(data)
    bout = b''.join(chunks)
    out = pkl.loads(bout)
    #print('Data: ',out)
    return out


def get_logs(fmjd:float, tmjd:float) -> str:
    """Get logs from database

    Args:
        fmjd (float): mjd from which logs should be taken
        tmjd (float): mjd to which logs should be taken

    Returns:
        str: response from database
    """

    res = dbquery( "select * from logs where" +
           " ( ( mjd>%f and mjd<%f) "%(fmjd,tmjd) +
           " or (mjd2>%f and mjd2<%f)"%(fmjd,tmjd) +
            " or (mjd<%f and mjd2>%f) "%(fmjd,tmjd) +
            " )  and new_id IS NULL ; ")
    return res


def get_err_logs(fmjd, tmjd, l):
    res = get_logs(fmjd,tmjd)
    for x in res:
        print(x[3].split('_')[0])
    return [x for x in res if 
                x[3].split('_')[0] == 'e' and
                len(set(l) & set(x[3].split('_'))) > 0]


def sendMessage(mjd, mjd2, mes, tag):
    s =( "INSERT INTO logs (mjd, mjd2, mes, tag) VALUES ("+
            "'"+mjd +"','"
            +mjd2+"','"+
            mes+"','"+tag+"');" )
    dbsend(s)


def modifyMessage( mjd,mjd2, mes,tag, prev_id):
    s =( "INSERT INTO logs (mjd, mjd2, mes, tag, prev_id) VALUES ("+
            "'"+mjd +"','"
            +mjd2+"','"+mes+"','"+tag+"','"+ prev_id  +  " ');" )
    i = dbsend(s)
    s = "UPDATE logs SET new_id = %d WHERE id = %s;"%(i,prev_id) 
    dbsend(s)


def dbsend(querstr):
    try:
        con = connect()
        cur=con.cursor()
        cur.execute(querstr)
        rowid = cur.lastrowid
        con.commit()
        cur.close()
        con.close()
        return rowid
    except:
        print('remote dbsend')
        return dbsend_rm(querstr)


def dbsend_rm(querstr):
    s = socket.socket()
    s.connect(('158.75.4.87', 12346))
    ss='dbs:'+querstr
    s.send(ss.encode())
    #receiving
    buffer = 1024
    received = 0
    chunks = []
    while received<4:
        data = s.recv(4-received)
        received += len(data)
        chunks.append(data)
    fsize = struct.unpack('!I',b''.join(chunks))[0]
    print('Size: ',fsize)

    received = 0
    chunks = []
    while received < fsize:
        data = s.recv(min(fsize-received, buffer)  )
        received += len(data)
        chunks.append(data)
    bout = b''.join(chunks)
    out = pkl.loads(bout)
    return out


def get_logs(fmjd, tmjd):
    res = dbquery( "select * from logs where" +
           " ( ( mjd>%f and mjd<%f) "%(fmjd,tmjd) +
           " or (mjd2>%f and mjd2<%f)"%(fmjd,tmjd) +
            " or (mjd<%f and mjd2>%f) "%(fmjd,tmjd) +
            " )  and new_id IS NULL ; ")
    return res


def dbsend_tmvl(tmvl):
    con = connect()
    cur=con.cursor()
    if isinstance(tmvl, list):
        for x in tmvl:
            print('----------------' ,x[0], x[1], x[2])
            s = "INSERT INTO %s (mjd,val) VALUES ('%s','%s');" % (
                    x[0], x[1], x[2])
            print(s)
            
            try:
                cur.execute(s)
            except db.Error as err:
                if err.errno == 1146:
                    print('No table '+x[0]+' in database')
                else:
                    raise
    else:
        print('Error: dbsend_tmvl: Argument is not a list')
    con.commit()
    cur.close()
    con.close()


def db_create_table(name):
    q = ("CREATE TABLE IF NOT EXISTS "+
            name+
            " (mjd DOUBLE NOT NULL, val DOUBLE NOT NULL);" )
    dbsend(q)


def rmerr(mts,l):
    mts.rmemptyseries()
    el = []
    if len(mts.dtab)>0 and len(mts.dtab[0].mjd_tab)>0:
        el = get_err_logs(mts.dtab[0].mjd_tab[0],
                          mts.dtab[-1].mjd_tab[-1],l)
    for x in el:
        mts.rmrange(x[0],x[1])
    return mts


def addDataToDatabase(tableName, mjd, val):
    db_create_table(tableName)
    con = connect()
    cur=con.cursor()
    cur.execute('INSERT INTO ' +tableName +
                ' VALUES ('+"%7.8f" % mjd + ',' +"%20.10f" % val +');' )
    con.commit()
    cur.close()
    con.close()
    

def addDataRangeToDatabase(tableName, fmjd, tmjd, val, step_mjd=2e-5):
    db_create_table(tableName)
    con = connect()
    cur=con.cursor()
    for mjd in np.arange(fmjd, tmjd, step_mjd):
        cur.execute('INSERT INTO ' +tableName +
                ' VALUES ('+"%7.8f" % mjd + ',' +"%20.10f" % val +');' )
        con.commit()
    cur.close()
    con.close()
