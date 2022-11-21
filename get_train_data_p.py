
from io import StringIO
import csv
import os
from collections import Counter
import datetime
import numpy as np
import pandas as pd
import shutil
from numpy import loadtxt
import warnings
warnings.filterwarnings("ignore")

################################################################################
#### get lab train gps
def get_train_gps( record_user, record_date):
    ###
    day_range   =   7
    ### 
    head = ["Account", "startlat", "startlng", "endlat", "endlng", "starttime", "endtime", "costtime",  "distance",  "formatetime",  "speed", "offline", "phone"]
    dir_train   = "train_p/"
    dir_split   = "./split_p/"
    dir_gps     = "gps/"
    # target date
    end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    # traget date
    start_date  = end_date - datetime.timedelta( days = day_range)
    # read data & reformat date
    try:
        #1
        dr =  pd.read_csv(dir_split + dir_gps + record_user + ".csv", header = None, skiprows=1)
        dr.columns = head
        #
        dateFormatter = "%Y-%m-%d %H:%M:%S"
        dr['formatetime'] = pd.to_datetime(dr['formatetime']).dt.strftime( dateFormatter)
        dr['formatetime'] = dr['formatetime'].astype('datetime64[ns]')
        ########
        # get data in target dates
        target = dr[( ( dr["formatetime"] > start_date) & 
                    (   dr["formatetime"] < end_date  ))]
        #### output train gps
        df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
        try:
            if not os.path.exists( df):
                os.makedirs( df)
        except OSError:
            print ('Error: Creating directory. ' +  df)
        target.to_csv( df + "/gps_raw.csv", index=False)
        print( df + " GPS complete")
    except (FileNotFoundError):
        pass
    return


#### get info
# 0=文字
# 1=語音
# 2=純情緒標記
# 3=Video
# 4=每日情緒
# 5=起床時間
# 6=每周自評量表
# 7=每周自評量表
# 8=睡覺時間

def get_train_info_date( tdr, record_user, record_date, types, idx_time):
    ###
    day_range   =   10
    ###
    dir_train   = "train_p/"
    # target date
    end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    # traget date
    start_date  = end_date - datetime.timedelta( days = day_range)
    # get target date gps
    target = tdr[( ( tdr[ idx_time] > start_date) & 
                   ( tdr[ idx_time] < end_date  ))]
    #### output train gps
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ########
    target.to_csv( df + "/info_"+ types +".csv", index=False)
    print( df + " info_" + types + " complete")
    return



def get_train_info( record_user, record_date):
    dir_split   = "./split_p/"
    dir_info    = "info/"
    try:
        dr =  pd.read_csv( dir_split + dir_info + "user/" + record_user + ".csv", header=None, skiprows=1, index_col=False)
        # reformat date
        idx_type = 1
        idx_time = 19
        dateFormatter = "%Y-%m-%d %H:%M:%S"
        dr[ idx_time] = pd.to_datetime(dr[ idx_time]).dt.strftime( dateFormatter)
        dr[ idx_time] = dr[ idx_time].astype('datetime64[ns]')
        # 0=文字
        tdr = dr[ dr[idx_type] == 0]
        get_train_info_date( tdr, record_user, record_date, "text", idx_time)
        # 1=語音
        tdr = dr[ dr[idx_type] == 1]
        get_train_info_date( tdr, record_user, record_date, "speech", idx_time)
        # 2=純情緒標記
        tdr = dr[ dr[idx_type] == 2]
        get_train_info_date( tdr, record_user, record_date, "emotion", idx_time)
        # 3=Video
        tdr = dr[ dr[idx_type] == 3]
        get_train_info_date( tdr, record_user, record_date, "video", idx_time)
        # 4=每日情緒
        tdr = dr[ dr[idx_type] == 4]
        get_train_info_date( tdr, record_user, record_date, "Demotion", idx_time)
        # 5=起床時間
        # 8=睡覺時間
        tdr = dr[ ( (dr[idx_type] == 5) | ( dr[idx_type] == 8))]
        get_train_info_date( tdr, record_user, record_date, "sleeptime", idx_time)
        # 6=每周自評量表
        # 7=每周自評量表
        tdr = dr[ ( (dr[idx_type] == 6) | ( dr[idx_type] == 7))]
        get_train_info_date( tdr, record_user, record_date, "selfscale", idx_time)
    except (FileNotFoundError):
        pass
    return



def get_train_target( record_user, record_date, data):
    ########
    end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ########
    # YMRS
    tmp     = pd.concat( [ data[ 3 : 14], data[ 38:39]])
    YMRS    = np.array( tmp, dtype=np.uint8)
    np.savetxt( df + "/target_ymrs.csv", YMRS, delimiter=",")
    # HAMD
    tmp     = pd.concat( [ data[ 14 : 38], data[ 39:40]])
    HAMD    = np.array( tmp, dtype=np.uint8)
    np.savetxt( df + "/target_hamd.csv", HAMD, delimiter=",")
    print( df + " Target complete")
    return 



def update_phones():  
    #######################################################################
    global altman
    global dass
    global account
    global id_user
    ### user name
    phones = np.full( ( len( dass)), 'NULL' , dtype=object)
    for i in range( len( dass)):
        name = dass[ 1].iloc[ i]
        name = name.lower().strip()
        idx = account.index[ account[ 'Name'].str.lower() == name].tolist()
        if len( idx) > 0:
            phone = account[ 'mobile'].iloc[ idx[0]]
            phones[ i] = phone
    ### user account
    for i in range( len( dass)):
        name = dass[ 1].iloc[ i]
        name = name.lower().strip()
        idx = account.index[ account[ 'Account'].str.lower() == name].tolist()
        if len( idx) > 0:
            phone = account[ 'mobile'].iloc[ idx[0]]
            phones[ i] = phone      
    ### user name
    for i in range( len( dass)):
        name = dass[ 1].iloc[ i]
        name = name.lower().strip()
        idx = id_user.index[ id_user[ 1].str.lower() == name].tolist()
        if len( idx) > 0:
            if phones[ i] == 'NULL':
                phone = id_user[ 2].iloc[ idx[0]]
                phones[ i] = phone
    ### add phones to DF
    tmp = pd.DataFrame( phones)
    dass[ 'phone'] = tmp
    #######################################################################
    ### user name
    phones = np.full( ( len( altman)), 'NULL' , dtype=object)
    for i in range( len( altman)):
        name = altman[ 1].iloc[ i]
        name = name.lower().strip()
        idx = account.index[ account[ 'Name'].str.lower() == name].tolist()
        if len( idx) > 0:
            phone = account[ 'mobile'].iloc[ idx[0]]
            phones[ i] = phone
    ### user account
    for i in range( len( altman)):
        name = altman[ 1].iloc[ i]
        name = name.lower().strip()
        idx = account.index[ account[ 'Account'].str.lower() == name].tolist()
        if len( idx) > 0:
            phone = account[ 'mobile'].iloc[ idx[0]]
            phones[ i] = phone 
    ### user name
    for i in range( len( altman)):
        name = altman[ 1].iloc[ i]
        name = name.lower().strip()
        idx = id_user.index[ id_user[ 1].str.lower() == name].tolist()
        if len( idx) > 0:
            if phones[ i] == 'NULL':
                phone = id_user[ 2].iloc[ idx[0]]
                phones[ i] = phone
    ### add phones to DF
    tmp = pd.DataFrame( phones)
    altman[ 'phone'] = tmp
    return



def read_selfscale():
    #
    dir_raw     = "./raw/"
    file_dass   = 'dass.csv'
    file_alt    = 'altman.csv'
    dateFormatter = "%Y/%m/%d"
    # DASS
    dass =  pd.read_csv( dir_raw + file_dass, header=None, skiprows=[0], index_col=False)
    idx_time = 0
    for i in range( len( dass)):
        #print( 'Dass  ' + str(i))
        tmp = dass[ idx_time][ i].split()[0]
        dass[ idx_time][ i] = datetime.datetime.strptime( tmp, dateFormatter)
    # Altman
    altman =  pd.read_csv( dir_raw + file_alt, header=None, skiprows=[0], index_col=False)
    idx_time = 0
    for i in range( len( altman)):
        #print( 'Altman  ' + str(i))
        tmp = altman[ idx_time][ i].split()[0]
        altman[ idx_time][ i] = datetime.datetime.strptime( tmp, dateFormatter)
    return dass, altman



def get_train_selfscale( dass, altman, record_user, record_date):
    ###
    day_range   = 10
    ###
    # get date and data
    # traget date
    end_date    = datetime.datetime.strptime( record_date, "%Y/%m/%d")
    # traget date
    start_date  = end_date - datetime.timedelta( days = day_range)
    #
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ############# DASS ############
    idx_time = 0
    idx_user = 'phone'
    # get user
    udr = dass[ dass[ idx_user] == record_user]
    # get after start time
    sdr = udr[ udr[ idx_time] > start_date]
    # get before end time
    edr = sdr[ sdr[ idx_time] < end_date]
    if len( edr) > 0:
        # get latest one scale
        want = edr.tail( 1)
        want.to_csv( df + "/selfscale_dass.csv", index=False)
        print( df + " DASS complete")
    ############# Altman ############
    idx_time = 0
    idx_user = 'phone'
    # get user
    udr = altman[ altman[ idx_user] == record_user]
    # get after start time
    sdr = udr[ udr[ idx_time] > start_date]
    # get before end time
    edr = sdr[ sdr[ idx_time] < end_date]
    # get last one
    if len( edr) > 0:
        want = edr.tail( 1)
        want.to_csv( df + "/selfscale_altman.csv", index=False)
        print( df + " Altman complete")
    return


####################################################################
####################################################################
####################################################################
####################################################################


### RUN HERE

##
file_gps    = "GPS.csv"
file_info   = "INFO.csv"
file_dass   = 'dass.csv'
file_alt    = 'altman.csv'


file_user_p = "id_user.xlsx"
file_hy_p   = "scale_p.csv"
file_account= "accounts.csv"

##
dir_raw_p   = "./raw/"
dir_patient = "p/"
dir_split   = "./split_p/"
dir_gps     = "gps/"
dir_info    = "info/"
dir_train   = "train_p/"

####################################################################

# create train directory
d = dir_train
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)


#### Read account
account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
account[ 'Account']  = account[ 'Account'].astype( str) 
account[ 'mobile']  = account[ 'mobile'].astype( str) 


#### Read user_id
id_user = pd.read_excel( dir_raw_p + dir_patient + file_user_p, header = None, dtype=str)
id_user[ 0] = id_user[ 0].astype( int)
id_user[ 1] = id_user[ 1].astype( str)
id_user[ 2] = id_user[ 2].astype( str)
for i in range( len( id_user)):
    id_user[ 1].iloc[ i] = id_user[ 1].iloc[ i].strip()


####################################################################

#### get self-report scale data
dass, altman =  read_selfscale()
dass[ 1]     =  dass[ 1].astype( str)
altman[ 1]   =  altman[ 1].astype( str)
update_phones()




####################################################################
#load model
import joblib

#load model
alltype_model=joblib.load('LA_alltype_model')
ABCD_model=joblib.load('LA_ABCD_model')
ABCE_model=joblib.load('LA_ABCE_model')
ABDE_model=joblib.load('LA_ABDE_model')
BCDE_model=joblib.load('LA_BCDE_model')
BDE_model=joblib.load('LA_BDE_model')
BCE_model=joblib.load('LA_BCE_model')
BCD_model=joblib.load('LA_BCD_model')


A = [ 'gps']
B = [ 'dass', 'altman']
C = [ 'Demo']
D = [ 'sleep']
E = [ 'media']
#####
# read HY scale
#scale_p = pd.read_csv( dir_raw_p + dir_patient + file_hy_p)
#all type ex:0932708935,2020/5/14
#ABCD : '0902393685_2020_09_04'
print('gogo enter')
u_date='0932708935,2020/5/14'
u_date='0902393685,2020/9/4'
#u_date='0933353413,2020/07/28'
check=1
while (u_date!='quit'):
    dt = loadtxt('data_type.csv',dtype='str', delimiter=',')
    dt = np.array(dt).tolist()
    #u_date=input()
    #delete train_p
    pathTest = "./train_p"
    try:
        shutil.rmtree(pathTest)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")
    #
    x=u_date.split(',')
    print('phone: ',x[0],' date:  ',x[1])
    user_p=x[0]
    date=x[1]
    #creating raw data by date
    get_train_selfscale( dass, altman, user_p, date)
    #get_train_target( user_p, date, scale_p[:].iloc[i])
    get_train_gps( user_p, date)
    get_train_info( user_p, date)
    #feature extraction    
    os.system("python feature_info_p.py")
    print('END feature_info')
    os.system("python feature_gps_p.py")
    print('END feature_gps')
    os.system("python multi_1030.py")
    print('END take feature')
    #load data and run
    tx = loadtxt('data.csv', delimiter=',')
    if(tx.size==180):#A+B+C+D+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=alltype_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==124):#A+B+C+D
        tx=np.reshape(tx,(1,tx.size))
        p_ty=ABCD_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==132 and A in dt):#A+B+C+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=ABCE_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==172):#A+B+D+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=ABDE_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==140):#B+C+D+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=BCDE_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==132):#B+D+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=BDE_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==92):#B+C+E
        tx=np.reshape(tx,(1,tx.size))
        p_ty=BCE_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    elif(tx.size==84):#B+C+D
        tx=np.reshape(tx,(1,tx.size))
        p_ty=BCD_model.predict(tx).astype(int)
        print('the predicted HAMD score of data is ',p_ty)
    else:
        print('[Error] Lack some data type!')
        if A not in dt:
            print('data type lack type ',A)
        if B not in dt:
            print('data type lack type ',B)
        if C not in dt:
            print('data type lack type ',C)
        if D not in dt:
            print('data type lack type ',D)
        if E not in dt:
            print('data type lack type ',E)
    u_date='quit'
    



