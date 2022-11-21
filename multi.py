
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures


from itertools import permutations
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

###################################################################################

def initial( weekly):
    #
    global altman
    global dass
    #
    global Demo
    #
    global gps
    #
    global sleep
    #
    global media
    #
    global hamd
    global ymrs
    #
    global f_shape
    global f_shape_w
    #
    shape       =   f_shape.copy()
    if 'dass' in weekly:
        shape[ 'dass']  =   f_shape_w[ 'dass']
    if 'altman' in weekly:
        shape[ 'altman']=   f_shape_w[ 'altman']
    if 'Demo' in weekly:
        shape[ 'Demo']  =   f_shape_w[ 'Demo']
    if 'sleep' in weekly:
        shape[ 'sleep'] =   f_shape_w[ 'sleep']
    if 'media' in weekly:
        shape[ 'media'] =   f_shape_w[ 'media']
    ### initial
    # 
    altman  = np.empty((0, shape[ 'altman']))
    dass    = np.empty((0, shape[ 'dass']))
    #
    Demo    = np.empty((0, shape[ 'Demo']))
    #
    gps     = np.empty((0, shape[ 'gps']))
    #
    sleep   = np.empty((0, shape[ 'sleep']))
    #
    media   = np.empty((0, shape[ 'media']))
    #
    hamd    = np.empty((0, shape[ 'hamd']))
    ymrs    = np.empty((0, shape[ 'ymrs']))
    return

###################################################################################

def neg_zero( L):
    for i in range( len(L)):
        if L[i] < 0:
            L[i] = 0
    return L 


def LG( train_x, train_y, test_x, test_y, neg):
    # sckit-learn implementation
    # Model initialization
    global PREDICT_VALUE
    regression_model = LinearRegression( n_jobs = 4)
    # Fit the data(train the model)
    regression_model.fit( train_x, train_y)
    # Predict
    y_predicted = regression_model.predict( test_x).astype(int)
    if neg:
        y_predicted = neg_zero( y_predicted)
    # model evaluation
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( test_y, y_predicted)
        rmae    = mean_absolute_error( test_y, y_predicted)
        rmse    = mean_squared_error( test_y, y_predicted)
        rves    = explained_variance_score( test_y, y_predicted)
        r2      = r2_score( test_y, y_predicted)
        print( r2)
    else:
        rmae    = metrics.f1_score( test_y, y_predicted, average='weighted')
        rmae    = metrics.accuracy_score( test_y, y_predicted)
        #rmae    = metrics.average_precision_score( test_y, y_predicted, average='samples')
        r2      = cs_score( test_y, y_predicted)
        rmse    = rmae
        # r2      = r2_score( test_y, y_predicted)
        print( r2)
    #
    x = abs( test_y - y_predicted)
    x = ['{:f}'.format(item) for item in x]
    ####
    # print("RMAE of test set is {}".format( round(rmae, 3)))
    # print("MSE of test set is {}".format( round(rmse, 3)))
    # print( test_y)
    # print( y_predicted)
    # print( x)
    return regression_model, rmae, rmse



def RG( train_x, train_y, test_x, test_y, neg):
    # sckit-learn implementation
    # Model initialization
    global alpha_
    global PREDICT_VALUE
    global tol
    if PREDICT_VALUE:
        ridge = Ridge( alpha = alpha_)
    else:
        ridge = RidgeClassifier( alpha = alpha_)
    # Fit the data(train the model)
    ridge.fit( train_x, train_y)
    # Predict
    y_predicted = ridge.predict( test_x).astype(int)
    if neg:
        y_predicted = neg_zero( y_predicted)
    # model evaluation
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( test_y, y_predicted)
        rmae    = mean_absolute_error( test_y, y_predicted)
        rmse    = mean_squared_error( test_y, y_predicted)
        rves    = explained_variance_score( test_y, y_predicted)
        r2      = r2_score( test_y, y_predicted)
        print( r2)
    else:
        # rmae    = metrics.f1_score( test_y, y_predicted, average='weighted')
        rmae    = metrics.accuracy_score( test_y, y_predicted)
        #rmae    = metrics.average_precision_score( test_y, y_predicted, average='samples')
        r2      = cs_score( test_y, y_predicted)
        rmse    = rmae
        # r2      = r2_score( test_y, y_predicted)
        print( r2)
    #
    x = abs( test_y - y_predicted)
    x = ['{:f}'.format(item) for item in x]
    ####
    # print("RMAE of test set is {}".format( round(rmae, 3)))
    # print("MSE of test set is {}".format( round(rmse, 3)))
    # print( test_y)
    # print( y_predicted)
    # print( x)
    print( len( ridge.coef_))
    return ridge, rmae, rmse



def LA( train_x, train_y, test_x, test_y, neg):
    # sckit-learn implementation
    # Model initialization
    global alpha_
    global PREDICT_VALUE
    global tol
    if PREDICT_VALUE:
        lasso = Lasso( alpha = alpha_)
    else:
        lasso = Lasso( alpha = alpha_, tol = tol)
    # Fit the data(train the model)
    lasso.fit( train_x, train_y)
    # Predict
    y_predicted = lasso.predict( test_x).astype(int)
    if neg:
        y_predicted = neg_zero( y_predicted)
    # model evaluation
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( test_y, y_predicted)
        rmae    = mean_absolute_error( test_y, y_predicted)
        rmse    = mean_squared_error( test_y, y_predicted)
        rves    = explained_variance_score( test_y, y_predicted)
        r2      = r2_score( test_y, y_predicted)
        print( r2)
    else:
        # rmae    = metrics.f1_score( test_y, y_predicted, average='weighted')
        rmae    = metrics.accuracy_score( test_y, y_predicted)
        #rmae    = metrics.average_precision_score( test_y, y_predicted, average='samples')
        r2      = cs_score( test_y, y_predicted)
        rmse    = rmae
        # r2      = r2_score( test_y, y_predicted)
        print( r2)
    #
    x = abs( test_y - y_predicted)
    x = ['{:f}'.format(item) for item in x]
    ####
    # print("RMAE of test set is {}".format( round(rmae, 3)))
    # print("MSE of test set is {}".format( round(rmse, 3)))
    # print( test_y)
    # print( y_predicted)
    # print( x)
    return lasso, rmae, rmse



def LOG( train_x, train_y, test_x, test_y, neg):
    # sckit-learn implementation
    # Model initialization
    global alpha_
    global l1_ratio_
    global PREDICT_VALUE
    global tol
    #
    if PREDICT_VALUE:
        LOG = LogisticRegression( penalty='elasticnet')
    else:
        LOG = LogisticRegression( penalty='elasticnet', tol=tol)
    # Fit the data(train the model)
    LOG.fit( train_x, train_y)
    # Predict
    y_predicted = LOG.predict( test_x).astype(int)
    if neg:
        y_predicted = neg_zero( y_predicted)
    # model evaluation
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( test_y, y_predicted)
        rmae    = mean_absolute_error( test_y, y_predicted)
        rmse    = mean_squared_error( test_y, y_predicted)
        rves    = explained_variance_score( test_y, y_predicted)
        r2      = r2_score( test_y, y_predicted)
        print( r2)
    else:
        # rmae    = metrics.f1_score( test_y, y_predicted, average='weighted')
        rmae    = metrics.accuracy_score( test_y, y_predicted)
        #rmae    = metrics.average_precision_score( test_y, y_predicted, average='samples')
        r2      = cs_score( test_y, y_predicted)
        rmse    = rmae
        # r2      = r2_score( test_y, y_predicted)
        print( r2)
    #
    x = abs( test_y - y_predicted)
    x = [ '{:f}'.format(item) for item in x]
    ####
    # print("RMAE of test set is {}".format( round(rmae, 3)))
    # print("MSE of test set is {}".format( round(rmse, 3)))
    print( test_y)
    print( y_predicted)
    # print( x)
    return LOG, rmae, rmse



def EN( train_x, train_y, test_x, test_y, neg):
    # sckit-learn implementation
    # Model initialization
    global alpha_
    global l1_ratio_
    global PREDICT_VALUE
    global tol
    #
    if PREDICT_VALUE :
        en = ElasticNet( alpha = alpha_, l1_ratio = l1_ratio_, fit_intercept = True)
    else:
        en = ElasticNet( alpha = alpha_, l1_ratio = l1_ratio_, tol = tol ,fit_intercept = True)
    # Fit the data(train the model)
    en.fit( train_x, train_y)
    # Predict
    y_predicted = en.predict( test_x).astype(int)
    if neg:
        y_predicted = neg_zero( y_predicted)
    # model evaluation
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( test_y, y_predicted)
        rmae    = mean_absolute_error( test_y, y_predicted)
        rmse    = mean_squared_error( test_y, y_predicted)
        rves    = explained_variance_score( test_y, y_predicted)
        r2      = r2_score( test_y, y_predicted)
        print( r2)
    else:
        # rmae    = metrics.f1_score( test_y, y_predicted, average='weighted')
        rmae    = metrics.accuracy_score( test_y, y_predicted)
        #rmae    = metrics.average_precision_score( test_y, y_predicted, average='samples')
        r2      = cs_score( test_y, y_predicted)
        rmse    = rmae
        # r2      = r2_score( test_y, y_predicted)
        print( r2)
    #
    x = abs( test_y - y_predicted)
    x = ['{:f}'.format(item) for item in x]
    ####
    # print("RMAE of test set is {}".format( round(rmae, 3)))
    # print("MSE of test set is {}".format( round(rmse, 3)))
    print( test_y)
    print( y_predicted)
    # print( x)
    return en, rmae, rmse


###################################################################################

def create_polynomial_regression_model( degree , X_train, Y_train, X_test, Y_test, neg):
    global PREDICT_VALUE
    "Creates a polynomial regression model for the given degree"
    poly_features = PolynomialFeatures( degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform( X_train)
    # fit the transformed features to Linear Regression
    poly_model = LinearRegression( n_jobs = 4)
    poly_model.fit( X_train_poly, Y_train)
    # predicting on test data-set
    y_test_predict = poly_model.predict( poly_features.fit_transform( X_test)).astype(int)
    if neg:
        y_test_predict = neg_zero( y_test_predict)
    # evaluating the model on training dataset
    # evaluating the model on test dataset
    if PREDICT_VALUE:
        # rmse    = fuzzy_cf ( Y_test, y_test_predict)
        rmse    = mean_squared_error( Y_test, y_test_predict)
        rmae    = mean_absolute_error( Y_test, y_test_predict)
        rves    = explained_variance_score( Y_test, y_test_predict)
        r2      = r2_score( Y_test, y_test_predict)
        print( r2)
    else:
        rmae    = metrics.accuracy_score( Y_test, y_test_predict)
        rmse    = rmae
    #
    x = abs( Y_test - y_test_predict)
    x = ['{:f}'.format(item) for item in x]
    ###
    # print( len( poly_model.coef_))
    # print("RMAE of test set is {}".format( round(rmae_test, 4)))
    # print("MSE of test set is {}".format( round(rmse, 4)))
    # print( Y_test)
    # print( y_test_predict)
    # print( x)
    return poly_model , rmae, rmse


###################################################################################
###################################################################################
###################################################################################
###################################################################################


def both_attend( a1, a2):
    ans = []
    flag=True
    for i in a1:
        if i in a2:
            ans.append( i)
    return ans

def find_attend( a1, a2):
    ans = []
    for i in a1:
        if i in a2:
            return True
        else:
            return False


def dif_attend( a1, a2):
    ans = []
    for i in a1:
        if not ( i in a2):
            ans.append( i)
    return ans

##1102
def get_real_who( who):
    print('enter_get_stack')
    #
    global all_attend
    global dass_attend
    global altman_attend
    global Demotion_attend
    global sleep_attend
    global media_attend
    #
    global gps_1000_attend
    global gps_4000_attend
    global gps_attend
    #
    ##
    tmp_who=[]
    tmp_ans = both_attend( all_attend, all_attend)
    if find_attend( tmp_ans, dass_attend)==True:
        tmp_who.append('dass')
    #
    if find_attend( tmp_ans, gps_attend)==True:
        tmp_who.append('gps')
    #
    if find_attend( tmp_ans, altman_attend)==True:
        tmp_who.append('altman')
    #
    if find_attend( tmp_ans, Demotion_attend)==True:
        tmp_who.append('Demo')
    #
    if find_attend( tmp_ans, sleep_attend)==True:
        tmp_who.append('sleep')
    #
    if find_attend( tmp_ans, media_attend)==True:
        tmp_who.append('media')
    ###########
    print('tmp_who',tmp_who)
    return tmp_who
##1102
### who => 想要的data type
## ans=> 哪幾份(week) 有 符合who
def get_attend( who):
    print('enter_get_stack')
    #
    global all_attend
    global dass_attend
    global altman_attend
    global Demotion_attend
    global sleep_attend
    global media_attend
    #
    global gps_1000_attend
    global gps_4000_attend
    global gps_attend
    #
    ans = both_attend( all_attend, all_attend)
    print('ans',ans)
    #
    if 'dass' in who:
        ans = both_attend( ans, dass_attend)
    #
    if 'altman' in who:
        ans = both_attend( ans, altman_attend)
    #
    if 'Demo' in who:
        ans = both_attend( ans, Demotion_attend)
    #
    if 'sleep' in who:
        ans = both_attend( ans, sleep_attend)
    #
    if 'media' in who:
        ans = both_attend( ans, media_attend)
    ###########
    ###########
    if 'gps' in who:
        ans = both_attend( ans, gps_attend)
    ##
    tmp_who=[]
    tmp_ans = both_attend( all_attend, all_attend)
    if find_attend( tmp_ans, dass_attend)==True:
        tmp_who.append('dass')
    #
    if find_attend( tmp_ans, altman_attend)==True:
        tmp_who.append('altman')
    #
    if find_attend( tmp_ans, Demotion_attend)==True:
        tmp_who.append('Demo')
    #
    if find_attend( tmp_ans, sleep_attend)==True:
        tmp_who.append('sleep')
    #
    if find_attend( tmp_ans, media_attend)==True:
        tmp_who.append('media')
    if find_attend( tmp_ans, gps_attend)==True:
        tmp_who.append('gps')
    ###########
    print('tmp_who',tmp_who)
    print('get ans',ans)
    return tmp_who,ans



def get_stack( who, rnd, weekly, normalized):
    #
    global altman
    global dass
    #
    global Demo
    #
    global gps
    #
    global sleep
    #
    global media
    #
    global hamd
    global ymrs
    #
    global f_shape
    global f_shape_w
    ###
    initial( weekly)
    who,attend = get_attend( who)
    #1102
    
    #1102
    stack  = np.empty( ( len(attend), 0))
    #
    global dir_
    dir_train = dir_
    #
    shape       =   f_shape.copy()
    f_gps       =   '/feature/gps_1000.csv'
    f_dass      =   '/feature/dass.csv'
    f_altman    =   '/feature/altman.csv'
    f_Demo      =   '/feature/Demotion.csv'
    f_sleep     =   '/feature/sleep.csv'
    f_media     =   '/feature/media.csv'
    #f_hamd      =   '/target_hamd.csv'
    #f_ymrs      =   '/target_ymrs.csv'
    if 'dass' in weekly:
        f_dass          =   '/feature/dass_w.csv'
        shape[ 'dass']  =   f_shape_w[ 'dass']
    if 'altman' in weekly:
        f_altman        =   '/feature/altman_w.csv'
        shape[ 'altman']=   f_shape_w[ 'altman']
    if 'Demo' in weekly:
        f_Demo          =   '/feature/Demotion_w.csv'
        shape[ 'Demo']  =   f_shape_w[ 'Demo']
    if 'sleep' in weekly:
        f_sleep         =   '/feature/sleep_w.csv'
        shape[ 'sleep'] =   f_shape_w[ 'sleep']
    if 'media' in weekly:
        f_media         =   '/feature/media_w.csv'
        shape[ 'media'] =   f_shape_w[ 'media']
    ###
    if rnd :
        np.random.shuffle( attend)
    #
    print('who ',who)
    for f in attend:
        # print( f)
        ########## get gps
        if 'gps' in who:
            tmp     = np.loadtxt( dir_train + f + f_gps, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'gps'])
            gps     = np.vstack( (gps, tmp))
        ########### get x_dass
        if 'dass' in who:
            tmp     = np.loadtxt( dir_train + f + f_dass, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'dass'])
            dass    = np.vstack( (dass, tmp))
        # get x_altman
        if 'altman' in who:
            tmp     = np.loadtxt( dir_train + f + f_altman, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'altman'])
            altman  = np.vstack( (altman, tmp))
        ########## get Demo
        if 'Demo' in who:
            tmp     = np.loadtxt( dir_train + f + f_Demo, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'Demo'])
            Demo    = np.vstack( (Demo, tmp))
        ########### # get sleep
        if 'sleep' in who:
            tmp     = np.loadtxt( dir_train + f + f_sleep, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'sleep'])
            sleep   = np.vstack( (sleep, tmp))
        ########### # get media
        if 'media' in who:
            tmp     = np.loadtxt( dir_train + f + f_media, delimiter=',')
            tmp     = np.reshape( tmp, shape[ 'media'])
            media   = np.vstack( (media, tmp))
        # get HAMD
        #tmp     = np.loadtxt( dir_train + f + f_hamd, dtype=np.float, delimiter=',')
        #hamd    = np.vstack( (hamd, tmp))
        # get YMRS
        #tmp     = np.loadtxt( dir_train + f + f_ymrs, dtype=np.float, delimiter=',')
        #ymrs    = np.vstack( (ymrs, tmp))
    #########################################################################
    ######
    ######
    if 'gps' in who:
        if normalized:
            normalized_gps()
        stack = np.hstack( ( stack, gps))
    ######
    ######
    if 'dass' in who:
        if normalized:
            if 'dass' in weekly:
                normalized_dass( True)
            else:
                normalized_dass( False)
        stack = np.hstack( ( stack, dass))
    #
    if 'altman' in who:
        if normalized:
            if 'altman' in weekly:
                normalized_altman( True)
            else:
                normalized_altman( False)
        stack = np.hstack( ( stack, altman))
    #
    if 'Demo' in who:
        if normalized:
            if 'Demo' in weekly:
                normalized_Demo( True)
            else:
                normalized_Demo( False)
        stack = np.hstack( ( stack, Demo))
    #
    if 'sleep' in who:
        if normalized:
            if 'sleep' in weekly:
                normalized_sleep( True)
            else:
                normalized_sleep( False)
        stack = np.hstack( ( stack, sleep))
    #
    if 'media' in who:
        if normalized:
            if 'media' in weekly:
                normalized_media( True)
            else:
                normalized_media( False)
        stack = np.hstack( ( stack, media))
    return attend, stack


def show( data):
    mean  = sum( data) / len( data)
    st    = np.std( data)
    # ±
    print( '%.2f ± %.1f'%( mean, st))
    return


def get_hamd( hamd, where, value):
    if where == 17:
        return label_hamd( np.sum( hamd[ : , 0:17], axis=1), value)
    elif where == 21:
        return label_hamd( np.sum( hamd[ : , 0:21], axis=1), value)
    elif where == 24: 
        return label_hamd( np.sum( hamd[ : , 0:24], axis=1), value)
    return label_hamd( hamd[:, -1], value)


# https://www.sciencedirect.com/science/article/pii/S0165032713003017?via%3Dihub
def label_hamd( hamd, value):
    if value:
        return hamd
    for i in range( len( hamd)):
        if hamd[ i] < 8:
            hamd[ i] = 1
        elif hamd[ i] < 17:
            hamd[ i] = 2
        elif hamd[ i] < 24:
            hamd[ i] = 3
        else:
            hamd[ i] = 4
    return hamd


def fuzzy_cf( test, predict):
    global FZ
    #
    test_l      = test.copy()
    predict_l   = predict.copy()
    test_l      = label_hamd( test_l, False)
    predict_l   = label_hamd( predict_l, False)
    #
    total = len( test)
    #
    c = 0
    for i in range( len( test)):
        if test_l[ i] == predict_l[ i]:
            c = c+1
        elif ( abs( test[ i] - predict[ i]) <= FZ):
            c = c+1
    return ( c/total)


def cs_score( y_test, y_predicted):
    #
    cm = confusion_matrix( y_test, y_predicted)
    #
    global pi
    global sv
    global sp
    pi  = []
    sv  = []
    sp  = []
    #
    for i in range( len( cm)):
        # precision
        u = cm[ i, i]
        d = sum( cm[:, i])
        if u==0 & d==0:
            pi.append( 1.0)
        else:
            pi.append( u/d )
        # recall
        u = cm[ i, i]
        d = sum( cm[ i, :])
        if u==0 & d==0:
            sv.append( 1.0)
        else:
            sv.append( u/d )
        # specificity
        tmp = cm.copy()
        tmp[ i, :] = 0
        d = sum( sum( tmp))
        tmp[ :, i] = 0
        u = sum( sum( tmp))
        if u==0 & d==0:
            sp.append( 1.0)
        else:
            sp.append( u/d )
    return np.array( (pi, sv, sp))


###################################################################################
###########################    ################  ##################################
###########################   #  ##############  ##################################
###########################   ###  ############  ##################################
###########################   #####  ##########  ##################################
###########################   #######  ########  ##################################
###########################   #########  ######  ##################################
###########################   ###########  ####  ##################################
###########################   #############  ##  ##################################
###########################   ###############    ##################################
###################################################################################


def normalized_gps():
    #
    global gps
    # [2, LV] [5, TD] [ 6, CST]
    idx_gps = [ 2, 5, 6, 7, 8, 9, 0, 1]
    gps_gap = [ 0, 10, 20, 30]
    # one x
    for x in range( len( gps)):
        # one feature
        for f in idx_gps:
            tmp = []
            # get one feature acorss group
            for g in gps_gap:
                tmp.append( gps[x][ g+f])
            max_ = max( tmp)
            min_ = min( tmp)
            # min max N
            for g in range( len( gps_gap)):
                if ( max_ - min_) == 0:
                    gps[ x][ gps_gap[g] + f] = 1
                else:
                    gps[ x][ gps_gap[g] + f] = ( tmp[g]-min_) / ( max_ - min_)
    return



def normalized_dass( week):
    #
    global dass
    #
    if week:
        #
        idx_dass_s  = 0
        # the maximum score of each scale
        max_dass    = 3 * 21
        min_dass    = 0
        ### dass
        # one x
        for x in range( len( dass)):
            # one feature
            dass[ x][ idx_dass_s] = ( dass[ x][ idx_dass_s]-min_dass) / ( max_dass - min_dass)
    else:
        # dass 21Q
        idx_dass    = list( range( 0, 21))
        idx_dass_s  = 21
        # the maximum score of each scale
        max_dass    = 3
        min_dass    = 0
        ### dass
        # one x
        for x in range( len( dass)):
            # one feature
            for f in idx_dass:
                dass[ x][ f] = ( dass[ x][ f]-min_dass) / ( max_dass - min_dass)
            dass[ x][ idx_dass_s] = ( dass[ x][ idx_dass_s]-min_dass) / ( max_dass*len( idx_dass) - min_dass)
    return


def normalized_altman( week):
    #
    global altman
    #
    if week:
        #
        idx_altman_s= 0
        # the maximum score of each scale
        max_altman  = 4 * 5
        min_altman  = 0
        ### altman
        # one x
        for x in range( len( altman)):
            # one feature
            altman[ x][ idx_altman_s] = ( altman[ x][ idx_altman_s]-min_altman) / ( max_altman - min_altman)
    else:
        # altman 5Q
        idx_altman  = list( range( 0 , 5))
        idx_altman_s= 5
        # the maximum score of each scale
        max_altman  = 4
        min_altman  = 0
        ### altman
        # one x
        for x in range( len( altman)):
            # one feature
            for f in idx_altman:
                altman[ x][ f] = ( altman[ x][ f]-min_altman) / ( max_altman - min_altman)
            altman[ x][ idx_altman_s] = ( altman[ x][ idx_altman_s]-min_altman) / ( max_altman*len( idx_altman) - min_altman)
    return



def normalized_Demo( week):
    #
    global Demo
    #
    if week:
        #
        max_    =  3
        # -1 to 1
        min_    =  0
        max_s   =  ( max_ - min_) / 2
        min_s   =  0
        #
        groups  = 4
        #
        for x in range( len( Demo)):
            idx = 0
            for g in range( groups):
                # mean
                Demo[x][ idx] = ( Demo[x][idx] - min_) / ( max_ - min_)
                idx = idx + 1
                # std
                Demo[x][ idx] = ( Demo[x][idx] - min_s) / ( max_s - min_s)
                idx = idx + 1
    else:
        #
        Demo_gap = [ 6, 7, 12, 13, 16, 17, 24, 25]
        #
        max_    =  3
        # -1 to 1
        min_    =  0
        max_s   =  ( max_ - min_) / 2
        min_s   =  0
        for x in range( len( Demo)):
            idx = 0
            while idx < len( Demo[ x]):
                if not idx in Demo_gap:
                    Demo[x][ idx] = ( Demo[x][idx] - min_) / ( max_ - min_)
                else:
                    # mean
                    Demo[x][ idx] = ( Demo[x][idx] - min_) / ( max_ - min_)
                    idx = idx + 1
                    # std
                    Demo[x][ idx] = ( Demo[x][idx] - min_s) / ( max_s - min_s)
                idx = idx + 1
    return



def normalized_sleep( week):
    #
    global sleep
    #
    if week:
        #####################
        groups    = 4
        # duration
        tmp_d     = []
        tmp_dt    = []
        # midpoint
        tmp_m     = []
        tmp_mt    = []
        # reg
        tmp_r     = []
        tmp_rt    = []
        #### get idx
        for d in range( groups):
            # d
            tmp_d.append(  0 + 12*d)
            tmp_dt.append( 2 + 12*d)
            # m
            tmp_m.append(  4 + 12*d)
            tmp_mt.append( 6 + 12*d)
            # r
            tmp_r.append(  8 + 12*d)
            tmp_rt.append( 10 + 12*d)
        #########################################
        for x in range( len( sleep)):
            idx = 0
            while idx < len( sleep[ x]):
                if idx in tmp_d:
                    max_ = 24
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_dt:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_m:
                    max_ = 48
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_mt:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_r:
                    max_ =  1
                    min_ =  0
                    # mean
                    idx = idx+1
                    # std
                    idx = idx+1
                elif idx in tmp_rt:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
    else:      
        #####################
        days  = [ 5, 4, 2, 5]
        # duration
        tmp_d     = []
        tmp_d_m   = []
        tmp_dt    = []
        tmp_dt_m  = []
        # midpoint
        tmp_m     = []
        tmp_m_m   = []
        tmp_mt    = []
        tmp_mt_m  = []
        # reg
        tmp_r     = []
        tmp_r_m   = []
        tmp_rt    = []
        tmp_rt_m  = []
        #### get idx
        idx = 0
        for d in days:
            # d
            for h in range( d):
                tmp_d.append( idx)
                idx = idx +1
            tmp_d_m.append( idx)
            idx = idx +1
            idx = idx +1
            # dt
            for h in range( d):
                tmp_dt.append( idx)
                idx = idx +1
            tmp_dt_m.append( idx)
            idx = idx +1
            idx = idx +1
            # m
            for h in range( d):
                tmp_m.append( idx)
                idx = idx +1
            tmp_m_m.append( idx)
            idx = idx +1
            idx = idx +1
            # mt
            for h in range( d):
                tmp_mt.append( idx)
                idx = idx +1
            tmp_mt_m.append( idx)
            idx = idx +1
            idx = idx +1
            # r
            for h in range( d-1):
                tmp_r.append( idx)
                idx = idx +1
            tmp_r_m.append( idx)
            idx = idx +1
            idx = idx +1
            # rt
            for h in range( d-1):
                tmp_rt.append( idx)
                idx = idx +1
            tmp_rt_m.append( idx)
            idx = idx +1
            idx = idx +1
        #########################################
        for x in range( len( sleep)):
            idx = 0
            while idx < len( sleep[ x]):
                if idx in tmp_d:
                    max_ = 24
                    min_ =  0
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                elif idx in tmp_dt:
                    max_ =  2
                    min_ =  0
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                elif idx in tmp_m:
                    max_ = 48
                    min_ =  0
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                elif idx in tmp_mt:
                    max_ =  2
                    min_ =  0
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                elif idx in tmp_r:
                    max_ =  1
                    min_ =  0
                    idx = idx+1
                elif idx in tmp_rt:
                    max_ =  2
                    min_ =  0
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                elif idx in tmp_d_m:
                    max_ = 24
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_dt_m:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_m_m:
                    max_ = 48
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_mt_m:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                elif idx in tmp_r_m:
                    max_ =  1
                    min_ =  0
                    # mean
                    idx = idx+1
                    # std
                    idx = idx+1
                elif idx in tmp_rt_m:
                    max_ =  2
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    # mean
                    sleep[x][ idx] = ( sleep[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                    # std
                    sleep[x][ idx] = ( sleep[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
    return


def normalized_media( week):
    #
    global media
    #
    if week:
        #####################
        # get idx
        tmp_m = [ 0, 2, 4, 6]
        tmp_s = [ 1, 3, 5, 7]
        #########################################
        len_emo = 11
        for x in range( len( media)):
            idx = 0
            while idx < ( len( media[ x])):
                tmp = int( idx / len_emo)
                # mean
                if tmp in tmp_m:
                    max_ =  3
                    min_ =  0
                    #
                    media[x][ idx] = ( media[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                # std
                elif tmp in tmp_s:
                    max_ =  3
                    min_ =  0
                    max_s= ( max_ - min_)/2 
                    min_s=  0
                    #
                    media[x][ idx] = ( media[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
    else:
        #####################
        days = [ 5, 4, 2, 5]
        tmp_m = [ ]
        tmp_s = [ ]
        #### get idx
        idx = 0
        for d in days:
            # m
            idx = idx + d
            tmp_m.append( idx)
            # s
            idx = idx + 1
            tmp_s.append( idx)
            #
            idx = idx + 1
        #########################################
        len_emo = 11
        for x in range( len( media)):
            idx = 0
            while idx < ( len( media[ x])):
                tmp = int( idx / len_emo)
                # mean
                if tmp in tmp_m:
                    max_ =  3
                    min_ =  0
                    #
                    media[x][ idx] = ( media[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
                # std
                elif tmp in tmp_s:
                    max_ =  3
                    min_ =  0
                    max_s= (max_ - min_)/2 
                    min_s=  0
                    #
                    media[x][ idx] = ( media[x][idx] - min_s) / ( max_s - min_s)
                    idx = idx+1
                # value
                else:
                    max_ =  3
                    min_ =  0
                    #
                    media[x][ idx] = ( media[x][idx] - min_) / ( max_ - min_)
                    idx = idx+1
    return





###################################################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
#########            #######  #####################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
#########  ########  #######  #####################################################
###################################################################################

def read_attend( dir_):
    #
    global all_attend
    global dass_attend
    global altman_attend
    global Demotion_attend
    global sleep_attend
    global media_attend
    #
    global gps_attend
    global gps_threshold
    #####
    tmp     = []
    for f in os.listdir( dir_):
        if os.path.isdir( dir_ + f):
            tmp.append( f)
    all_attend      = np.array( tmp)
    #gps_attend      = np.loadtxt( dir_train + "gps_attend.csv", dtype=np.str, delimiter=',')
    altman_attend     = np.loadtxt( dir_ + "altman_attend.csv", dtype=np.str, delimiter=',')
    dass_attend       = np.loadtxt( dir_ + "dass_attend.csv", dtype=np.str, delimiter=',')
    Demotion_attend   = np.loadtxt( dir_ + "Demotion_attend.csv", dtype=np.str, delimiter=',')
    sleep_attend      = np.loadtxt( dir_ + "sleep_attend.csv", dtype=np.str, delimiter=',')
    media_attend      = np.loadtxt( dir_ + "media_attend.csv", dtype=np.str, delimiter=',')
    #
    gps_attend   = np.loadtxt( dir_ + "gps_attend"+ str( gps_threshold) +".csv", dtype=np.str, delimiter=',')
    print('all_at: ',all_attend,' altman_at: ',altman_attend,' dass_at: ',dass_attend)
    return

# 確保dara是跟who的交集
def inter2all( a, x, h, y, C5):
    #
    a_all = []
    x_all = np.empty( ( 0, x.shape[1]))
    h_all = np.empty( ( 0, h.shape[1]))
    y_all = np.empty( ( 0, y.shape[1]))
    #
    for idx, user in enumerate( a):
        if user in C5:
            # a
            a_all.append( user)
            # x
            x_all = np.vstack(( x_all, x[idx]))
            # h
            h_all = np.vstack(( h_all, h[idx]))
            # y
            y_all = np.vstack(( y_all, y[idx]))
    return a_all, x_all, h_all, y_all



def make_feature_name( WEEK):
    #
    gps = []
    for i in range( 4):
        G = 'gps_G' + str( i+1) + '_'
        #
        gps.append( G + 'etp')
        gps.append( G + 'etp_n')
        gps.append( G + 'loc_var')
        gps.append( G + 'home_stay')
        gps.append( G + 'trans_time')
        gps.append( G + 'dis_all')
        gps.append( G + 'cst_num')
        gps.append( G + 'cir')
        gps.append( G + 'cir_N')
        gps.append( G + 'cir_home')
    #
    dass = []
    if 'dass' not in WEEK:
        for i in range( 21):
            tmp = 'dass_Q' + str( i+1)
            dass.append( tmp)
    dass.append( 'dass_total')
    #
    altman = []
    if 'altman' not in WEEK:
        for i in range( 5):
            tmp = 'altman_Q' + str( i+1)
            altman.append( tmp)
    altman.append( 'altman_total')
    #
    Demo = []
    days = [6, 4, 2, 6]
    for i in range( 4):
        G = 'Demo_G' + str( i+1) + '_'
        #
        if 'Demo' not in WEEK:
            for j in range( days[i]):
                Demo.append( G + 'v' + str( j+1))
        Demo.append( G + 'mean')
        Demo.append( G + 'std')
    #
    sleep = []
    days = [5, 4, 2, 5]
    for i in range( 4):
        G = 'sleep_G' + str( i+1) + '_'
        # duration
        if 'sleep' not in WEEK:
            for j in range( days[i]):
                sleep.append( G + 'dr_' + str( j+1))
        sleep.append( G + 'dr_mean')
        sleep.append( G + 'dr_std') 
        if 'sleep' not in WEEK:
            for j in range( days[i]):
                sleep.append( G + 'drt_' + str( j+1))
        sleep.append( G + 'drt_mean')
        sleep.append( G + 'drt_std') 
        # midpnts
        if 'sleep' not in WEEK:
            for j in range( days[i]):
                sleep.append( G + 'mp_' + str( j+1))
        sleep.append( G + 'mp_mean')
        sleep.append( G + 'mp_std') 
        if 'sleep' not in WEEK:
            for j in range( days[i]):
                sleep.append( G + 'mpt_' + str( j+1))
        sleep.append( G + 'mpt_mean')
        sleep.append( G + 'mpt_std') 
        # rg
        if 'sleep' not in WEEK:
            for j in range( days[i]-1):
                sleep.append( G + 'rg_' + str( j+1))
        sleep.append( G + 'rg_mean')
        sleep.append( G + 'rg_std') 
        if 'sleep' not in WEEK:
            for j in range( days[i]-1):
                sleep.append( G + 'rgt_' + str( j+1))
        sleep.append( G + 'rgt_mean')
        sleep.append( G + 'rgt_std') 
    #
    media = []
    days = [5, 4, 2, 5]
    for i in range( 4):
        G = 'media_G' + str( i+1) + '_'
        #
        if 'media' not in WEEK:
            for j in range( days[i]):
                for k in range( 11):
                    media.append( G + 'd' + str( j+1) + '_v' + str( k+1))
        for k in range( 11):
            media.append( G + 'mean_' + str( k+1))
        for k in range( 11):
            media.append( G + 'std_' + str( k+1))
    #
    return gps, dass, altman, Demo, sleep, media



def get_feature_name( who, WEEK):
    #
    gps, dass, altman, Demo, sleep, media = make_feature_name( WEEK)
    #
    f = []
    #
    for t in who:
        if (t == 'gps'):
            f = f + gps
        #
        if t == 'dass':
            f = f + dass
        #
        if t == 'altman':
            f = f + altman
        #
        if t == 'Demo':
            f = f + Demo
        #
        if t == 'sleep':
            f = f + sleep
        #
        if t == 'media':
            f = f + media
    return f


###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

##
dir_raw     = "./raw/"
dir_train_p = "train_p/"
dir_train=dir_train_p

## feature dimension
global f_shape
f_shape ={
    'gps'       : 40,
    'altman'    : 6,##(5 ques + sum)
    'dass'      : 22,
    'Demo'      : 26,
    'hamd'      : 25,
    'ymrs'      : 12,
    'sleep'     : 136,
    'media'     : 168,
    #'media'     : 264,
}

## feature dimension (WEEKLY)
global f_shape_w
f_shape_w ={
    'gps'       : 40,
    'altman'    : 1,##(only total)
    'dass'      : 1,
    'Demo'      : 8,
    'hamd'      : 25,
    'ymrs'      : 12,
    'sleep'     : 48,
    'media'     : 56,
    #'media'     : 88,
}


###
tmp     = []
for f in os.listdir( dir_train):
    if os.path.isdir( dir_train + f):
        tmp.append( f)


all_attend        = np.array( tmp)
#gps_attend      = np.loadtxt( dir_train + "gps_attend.csv", dtype=np.str, delimiter=',')
altman_attend     = np.loadtxt( dir_train + "altman_attend.csv", dtype=np.str, delimiter=',')
dass_attend       = np.loadtxt( dir_train + "dass_attend.csv", dtype=np.str, delimiter=',')
Demotion_attend   = np.loadtxt( dir_train + "Demotion_attend.csv", dtype=np.str, delimiter=',')
sleep_attend      = np.loadtxt( dir_train + "sleep_attend.csv", dtype=np.str, delimiter=',')
media_attend      = np.loadtxt( dir_train + "media_attend.csv", dtype=np.str, delimiter=',')


### determine the gps threshold
gps_threshold = 2000
gps_attend   = np.loadtxt( dir_train + "gps_attend" + str( gps_threshold) + ".csv", dtype=np.str, delimiter=',')


#
#global all_attend
#s
global altman
global dass
#
global Demo
#
global gps
#
global sleep
#
global media
#
global hamd
global ymrs


## who
gps_threshold = 2000
A = [ 'gps']
B = [ 'dass', 'altman']
C = [ 'Demo']
D = [ 'sleep']
E = [ 'media']


#rand five fold=>打散
rand    = True
Normal  = True
TF      = [ False, True]


###################################################################################

##### get all attend
who      = A + B + C + D + E
who=get_real_who(who)
print('real who',who)
WEEK     = C + D + E

##### patient
dir_    =   dir_train_p
read_attend( dir_)
print('all attend: ',all_attend)
#
a_C5_p, x_p = get_stack( who, rand, WEEK, Normal)
print(x_p.shape)
np.savetxt("data.csv",x_p,delimiter=",")
#save data type
rf=open("data_type.csv",'w')
for r in who:
    rf.write(r+',')
rf.close()    


