import MySQLdb as db
import numpy as np
import time_tools as tit
import socket
import pickle as pkl
import struct

def connect():
    """Connect to MySQL database on corridor    
    Returns:
        handler to connection
    """
    return db.connect( host='localhost', user='', password='', db='measurements')

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
