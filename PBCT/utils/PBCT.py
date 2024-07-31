from tkinter import FALSE
import pandas as pd
from SFS import Sequential_Forward_Selection_corr_test
from sklearn.linear_model import LinearRegression
from solve_loss import solve_loss
from sklearn.model_selection import LeaveOneOut,cross_val_score
import numpy as np
import math
import sklearn
import random
import matplotlib.pyplot as plt
import os


def PBCT(L,random_index,repeated_num,coef_file,data_labeled,data_unlabeled,data_test,data_columnslabel_x,data_columnslabel_y):
    """
    L:number of labeled data
    random_index: list of current index sample
    repeat_num: random experiment index (for repeated experiment)
    coef_file: file to store the coef of learned model parameter
    data_labeled: labeled data (feature and cyclelife)
    data_unlabeled: unlabeled data (feature and cyclelife(ignored))
    data_test: test data (feature and cyclelife)
    data_columnslabel_x: column names of feature
    data_columnslabel_y: column names of cyclelife

    """

    data_labeled_x = data_labeled[data_columnslabel_x]
    data_labeled_y = data_labeled[data_columnslabel_y]
    data_unlabeled_x = data_unlabeled[data_columnslabel_x]
    data_unlabeled_y = data_unlabeled[data_columnslabel_y]
    data_test_x = data_test[data_columnslabel_x]
    data_test_y = data_test[data_columnslabel_y]
    ####Nomalize####
    mean_labeled_x = data_labeled_x.mean()
    std_labeled_x = data_labeled_x.std()

    mean_labeled_y = data_labeled_y.mean()
    std_labeled_y = data_labeled_y.std()

    X_train_labeled = (data_labeled_x-mean_labeled_x)/std_labeled_x
    X_train_unlabeled = (data_unlabeled_x-mean_labeled_x)/std_labeled_x
    y_train = (data_labeled_y-mean_labeled_y)/std_labeled_y
    X_test = (data_test_x-mean_labeled_x)/std_labeled_x
    y_test = (data_test_y-mean_labeled_y)/std_labeled_y
    #################

    ####build Partial model####
    Partial_feature_var2 = Sequential_Forward_Selection_corr_test(data_columnslabel_x,data_labeled_x,data_labeled_y,random_index,L,L-2)
    Partial_feature = Partial_feature_var2[0]
    print(Partial_feature)


    ZL = data_labeled_x[Partial_feature]
    ZU = data_unlabeled_x[Partial_feature]
    Z_test = X_test[Partial_feature]
    #var2 = max(Partial_feature_var2[1],1e-3)
    var2 = Partial_feature_var2[1]
    print('var2 is ',var2)



    ####find the var1####
    V = [0.5,1,2,5,10]
    print('current V',V)
    var1_candidate_set = [i*var2 for i in V]

    l5_candidate = [10,100]

    LOO_list = []

    for i in range(len(V)):
        for m in range(len(l5_candidate)):
            tmp_var1 = var1_candidate_set[i]
            l1 = 1/2/tmp_var1
            l2 = 1/2/var2
            l3 = 0
            l4 = 1/2/(tmp_var1+var2)
            #l5 = 0
            l5 = l1*l5_candidate[m]
            error_list = []
            for j in range(L):
                predict_x = data_labeled_x.iloc[j]
                tmp_X = data_labeled_x.drop(random_index[j])
                tmp_ZL = ZL.drop(random_index[j])
                predict_y = data_labeled_y.iloc[j]
                tmp_y = data_labeled_y.drop(random_index[j])

                ##Normalize##
                mean_labeled_x = tmp_X.mean()
                std_labeled_x = tmp_X.std()
                mean_labeled_y = tmp_y.mean()
                std_labeled_y = tmp_y.std()
                mean_labeled_z = tmp_ZL.mean()
                std_labeled_z = tmp_ZL.std()

                X_train_tmp = (tmp_X - mean_labeled_x)/std_labeled_x
                y_train_tmp = (tmp_y-mean_labeled_y)/std_labeled_y
                X_train_unlabeled_tmp = (data_unlabeled_x-mean_labeled_x)/std_labeled_x

                tmp_ZL = (tmp_ZL-mean_labeled_z)/std_labeled_z
                tmp_ZU = (ZU-mean_labeled_z)/std_labeled_z


                predict_x = (predict_x-mean_labeled_x)/std_labeled_x


                alpha,beta = solve_loss(y_train_tmp.to_numpy().ravel(),X_train_tmp.to_numpy(),tmp_ZL.to_numpy(),
                        X_train_unlabeled_tmp.to_numpy(),tmp_ZU.to_numpy(),l1,l2,l3,l4,l5)

                real_predict_y = predict_y.to_numpy()
                alpha_y = np.matmul(alpha.T,predict_x.to_numpy())
                real_alpha_y = (alpha_y*std_labeled_y+mean_labeled_y).to_numpy()

                tmp_error = (real_predict_y[0] - real_alpha_y[0])
                tmp_error_square = tmp_error * tmp_error
                error_list.append(tmp_error_square)
            LOO_list.append(np.mean(error_list))
    var1_index = np.argmin(np.array(LOO_list))
    l1_index = var1_index // len(l5_candidate)
    l5_index = var1_index % len(l5_candidate)
    #######################
    print('var1_index',var1_index)
    ####get the optimal alpha and beta####
    l1 = 1/2/var1_candidate_set[l1_index]
    l2 = 1/2/var2
    l3 = 0
    l4 = 1/2/(var1_candidate_set[l1_index]+var2)
    l5 = l1*l5_candidate[l5_index]

    ZL_all = X_train_labeled[Partial_feature]
    ZU_all = X_train_unlabeled[Partial_feature]



    alpha,beta = solve_loss(y_train.to_numpy().ravel(),X_train_labeled.to_numpy(),ZL_all.to_numpy(),
                        X_train_unlabeled.to_numpy(),ZU_all.to_numpy(),l1,l2,l3,l4,l5)
    print('alpha is ', alpha)
    print('beta is', beta)
    Loss = (l1*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(X_train_labeled.to_numpy(),alpha)),2)+
            l2*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(ZL.to_numpy(),beta)),2)+
            l3*math.pow(np.linalg.norm(np.matmul(X_train_labeled.to_numpy(),alpha)-np.matmul(ZL.to_numpy(),beta)),2)+
            l4*math.pow(np.linalg.norm(np.matmul(X_train_unlabeled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #check_loss(y_train,X_train_labeled,ZL,
    #                    X_train_unlabeled,ZU,l1,l2,l3,l4,alpha,beta,Loss)
    print('Loss',math.pow(np.linalg.norm(np.matmul(X_train_unlabeled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #####################################


    ####start to test######
    test_err_list = []
    err_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labeled_y+mean_labeled_y
        tmp_pre = np.matmul(alpha.T,X_test.iloc[i].to_numpy())*std_labeled_y+mean_labeled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test_err_list.append(tmp_error_square)
        err_percent_list.append(tmp_percent_err)

    print('PBCT_RMSE',np.sqrt(np.mean(test_err_list)))
    print('PBCT_ERR',np.mean(err_percent_list))
    
    PBCT_RMSE = np.sqrt(np.mean(test_err_list))
    PBCT_ERR = np.mean(err_percent_list)
    #print(LOO_list)
    test0_err_list = []
    err0_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labeled_y+mean_labeled_y
        tmp_pre = np.matmul(beta.T,Z_test.iloc[i].to_numpy())*std_labeled_y+mean_labeled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test0_err_list.append(tmp_error_square)
        err0_percent_list.append(tmp_percent_err)

    print('PBCT_beta_RMSE',np.sqrt(np.mean(test0_err_list)))
    print('PBCT_beta_ERR',np.mean(err0_percent_list))
    PBCT_beta_RMSE = np.sqrt(np.mean(test0_err_list))
    PBCT_beta_ERR = np.mean(err0_percent_list)


    coef_list = []
    coef_list.append(np.array([v for v in alpha.ravel()]))
    coef_list.append(np.array(beta.ravel()))
    tmp_df = pd.DataFrame(data = coef_list)
    tmp_df.to_csv(coef_file+str(L)+'_'+str(repeated_num)+'.csv',index = False, header = False)
    return PBCT_RMSE, PBCT_beta_RMSE


if __name__=="__main__":
    coef_file = './tmp_coef/'
    if not os.path.exists(coef_file):
        os.makedirs(coef_file)
    repeated_num = 50
    all_average = {}
    all_median = {}
    labeled_data_csv_name = r'D:\WangQiao\labeled_data.xlsx'
    unlabeled_data_csv_name = r'D:\WangQiao\unlabeled_data.xlsx'
    labeled_data_samples = pd.read_excel(labeled_data_csv_name,index_col=0)
    unlabeled_data_samples = pd.read_excel(unlabeled_data_csv_name,index_col=0)

    file_name = 'tmp_file'
    Upper_var1 = 1

    labeled_num = len(labeled_data_samples)-1 # 标记样本总数
    unlabeled_num = len(unlabeled_data_samples)-1 # 无标记样本总数
    test_num = unlabeled_num # 测试样本总数
    data_num = labeled_num + unlabeled_num  # 样本总数

    # 生成一个包含 repeated_num 个元素的列表，每个元素是一个长度为 data_num 的随机排列列表。
    random_index = [random.sample(range(labeled_num),labeled_num) for i in range(repeated_num)]
    # random_index = labeled_data_samples.index.values.tolist()
    repeated_num = 0


    data_columnslabel_x = labeled_data_samples.columns[:-1] # 除去最后1列
    data_columnslabel_y = labeled_data_samples.columns[-1:] # 包含最后1列

    # train_index = random_index[repeated_num][:data_num-test_num]
    # test_index = random_index[repeated_num][data_num-test_num:]
    data_labeled = labeled_data_samples
    data_unlabeled = unlabeled_data_samples
    data_test = unlabeled_data_samples
    PBCT_RMSE, PBCT_beta_RMSE= PBCT(labeled_num, random_index[repeated_num],repeated_num,coef_file,
                                    data_labeled,data_unlabeled,data_test,data_columnslabel_x,data_columnslabel_y)
