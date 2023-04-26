import pandas as pd
import numpy as np

def user_input(input_chemical,pec_df,df2,stp_re,df_s,kow_input):

    chemical_i_want=input_chemical
    chemical=chemical_i_want
    t_df_pec = pec_df
    t_df_2 = df2
    stp_removal=stp_re
    df=df_s
    kow_value=kow_input

    N_FCODE3 = np.array([11479,11193,11707,11524,11162,11252,11289,11255,11423,11348,11690,11420]).astype('str') #물고기 코드
    N_FCODE3 = np.sort(N_FCODE3)
    weight = np.array([0.003,0.01,0.003,0.003,1,1,1,0.01,5,3,0.5,0.01]) # 물고기 weight
    Kow_list = [np.log10(float(kow_value))] # 입력한 화학물질 값에 따라서 KOW 불러옴

    T = 10                                              # Mean water temperature (C)
    L_b = 0.2                                           # Lipid content of organism (%)
    L_d = 0.01                                          # Lipid content of lowest trophic level organism (%)
    xPOC = 5*1e-7                                    # Concentration of particulate organic carbon (g/ml)
    xDOC = 5*1e-7                                    # Concentration of dissolved organic carbon (g/ml)
    beta = 130                                          # Overall food web biomagnification factor
    kM = 0                                              # Metabolic transformation rate constant (/d)
    n = 3                                               # Number of trophic interaction in the food web


    def kow(i):
        Kow = i
        W = weight   # variable                                       # Weight of organism  (kg)
        pi = 1/(1 + (xPOC*0.35*(10**Kow)) + (xDOC*0.1*0.35*(10**Kow)))    # Fraction of freely dissolved chemical in water
        tau = (0.0065/(kM + 0.0065))**(n-1)                # Maximum trophic dillution factor
        k1 = 1/((0.01 + (1/(10**Kow)))*(W**0.4))            # Uptake rate constant (L/kg*d)
        kD = (0.02 * (W**(-0.15)) * np.exp(0.06*T)) / (5.1*(10**(-8))*(10**Kow) + 2)
                                                        # Dietary Uptake rate constant (kg/kg*d)
        k2 = k1/(L_b*(10**Kow))                            # Elimination rate constant (/d)
        kE = 0.125*kD                                     # Fecal egestion rate constant (/d)
        kG = 0.0005*(W**(-0.2))                            # Growth rate constant (/d)
        
        BAF = (1-L_b) + ((k1*pi + (kD*beta*tau*pi*L_d*(10**Kow))) / (k2 + kE + kG + kM))
        BCF = (1-L_b) + (k1*pi/(k2 + kE + kG + kM))
        return BAF , BCF
    result = np.array(list( map(kow, Kow_list) ))
    BAFs= result[:,0] # chemcials x fishes
    #%% PEC x BAF 3 * 12 * 41 (chemicals x fishes x streams)
    def fish(i):
        BAF = BAFs[i]
        BAF = BAF.reshape((len(weight),1))
        PEC = np.array(t_df_pec.iloc[i]).reshape(1, len(t_df_2))
        fish_con = np.matmul(BAF,PEC)
        return fish_con
    fish_con_df =list( map(fish, np.arange(0,1)) )

    #%% df 합치기 
    def mapping(i):
        c = pd.DataFrame(fish_con_df[i], index = N_FCODE3, columns = range(0,len(t_df_2)))
        c.insert(0, 'chemical', df.index[i])
        return c
    c1 = pd.concat(map(mapping,  np.arange(0,1) ))


    #%%
    condition = (c1.chemical == chemical_i_want) # 조건식 작성 e.g., # 5836-29-3 불러내기

    con_i_want = c1[condition].drop(['chemical'], axis=1)

    #%% 음용수 농도 
    drinking_water = t_df_pec.mul(1-stp_removal/100, axis = 0)
    drinking_water_max = drinking_water.max(axis=1)
    # In[]

    # IR * C / BW
    # 데이터 불러오기 및 전처리

    # dir_path = os.chdir("C:/Users/gwyoo")

    intake_db = pd.read_csv('intake.csv', encoding = 'CP949')
    intake_db = intake_db.fillna(0)
    intake_db = intake_db.astype({'N_FCODE3':'int'})
    intake_db = intake_db.astype({'N_FCODE3':'str'})
    # HE_wt : 체중
    # N_WAT_C : 
    # intake.w_liter.per.day : 물섭취량
    # N_FCODE3 : 식품 코드
    # N_FNAME3 : 식품 이름
    # T_NF_INTK3 : 식품 섭취량


    T_df = pd.DataFrame(index=np.arange(len(intake_db)),columns = N_FCODE3).fillna(0.0)

    for k in range(len(T_df.columns)):
        for i in range(len(T_df)):
            if intake_db['N_FCODE3'][i] == T_df.columns[k]:
                T_df[T_df.columns[k]][i] = intake_db['T_NF_INTK3'][i]

    # 농도자료
    con_i_want_T = con_i_want.T

    drinking_water_T = drinking_water.T
    drinking_water_T = pd.DataFrame(drinking_water_T[chemical])
    # In[]
    # 500회 반복하여 어패류 농도 샘플링

    C_array1 = np.zeros(shape=(12,1))
    for h in range(500):
        C_array0 = np.array([])
        for i in range(len(con_i_want_T.columns)):
            C_array0 = np.append( C_array0, np.random.choice(con_i_want_T.iloc[:,i], 1).item() )
        C_array1 = np.hstack( (C_array1, C_array0.reshape(12,1)) )
        
    Cw_array1 = np.zeros(shape=(1,1))
    for h in range(500):
        Cw_array0 = np.array([])
        for i in range(len(drinking_water_T.columns)):
            Cw_array0 = np.append( Cw_array0, np.random.choice(drinking_water_T.iloc[:,i], 1).item() )
        Cw_array1 = np.hstack( (Cw_array1, Cw_array0.reshape(1,1)) )

    C_fr = C_array1[:,1:]
    Cw_wr = Cw_array1[:,1:]

    step2_idx = [3,8,9,11]
    step3_idx = [1,2,4,5,6,7,10]
    step4_idx = [0]

    t_df_3=t_df_2.reset_index()
    a1 = t_df_3['중권역명']
    a2=  t_df_3['STP명']
    a3 = con_i_want_T.iloc[:,step2_idx].sum(axis=1)
    a4 = con_i_want_T.iloc[:,step3_idx].sum(axis=1)
    a5 = con_i_want_T.iloc[:,step4_idx].sum(axis=1)
    aw = drinking_water_T.sum(axis=1)

    material_conc = pd.concat([a1,a2,aw,a3,a4,a5],axis=1).reset_index(drop=True)
    material_conc.columns = ['중권역명','STP명','음용수(µg/L)','영양단계2(µg/g)','영양단계3(µg/g)','영양단계4(µg/g)']
    material_conc=material_conc.set_index('중권역명')

    # In[]
    Con_2 = con_i_want_T.iloc[:,step2_idx].mean().mean()
    Con_3 = con_i_want_T.iloc[:,step3_idx].mean().mean()
    Con_4 = con_i_want_T.iloc[:,step4_idx].mean().mean()
    Con_w = drinking_water_T.mean().item()

    Con_2=format(Con_2,'.1E')
    Con_3=format(Con_3,'.1E')
    Con_4=format(Con_4,'.1E')
    Con_w=format(Con_w,'.1E')


    C_fr2 = C_fr[step2_idx,:]
    C_fr3 = C_fr[step3_idx,:]
    C_fr4 = C_fr[step4_idx,:].reshape(len(step4_idx), C_fr.shape[1])


    T_if = np.array(T_df)

    T_if2 = T_if[:,step2_idx]
    T_if3 = T_if[:,step3_idx]
    T_if4 = T_if[:,step4_idx].reshape(len(intake_db),1)

    Tw_iw = np.array(intake_db['intake.w_liter.per.day']).reshape(len(intake_db),1)

    TC_ir2 = np.matmul(T_if2, C_fr2)
    TC_ir3 = np.matmul(T_if3, C_fr3)
    TC_ir4 = np.matmul(T_if4, C_fr4)
    TCw_ir = np.matmul(Tw_iw, Cw_wr)

    Exp2_ir = TC_ir2 / np.array(intake_db['HE_wt']).reshape(len(intake_db),1)
    Exp3_ir = TC_ir3 / np.array(intake_db['HE_wt']).reshape(len(intake_db),1)
    Exp4_ir = TC_ir4 / np.array(intake_db['HE_wt']).reshape(len(intake_db),1)
    Expw_ir = TCw_ir / np.array(intake_db['HE_wt']).reshape(len(intake_db),1)
    Exp_ir = (TC_ir2 + TC_ir3 + TC_ir4 + TCw_ir) / np.array(intake_db['HE_wt']).reshape(len(intake_db),1)

    # In[]
    # 500개의 평균값
    Exp_r_mean = Exp_ir.mean(axis=0)
    np.quantile(Exp_r_mean, 0.05)
    np.quantile(Exp_r_mean, 0.95)

    # In[]

    #엑셀에서 어패류중 민물 어패류만 확인 후 담수 어패류 행 추가
    #엑셀에서 FCR2,3,4 단계 확인후 표기 FCR2 : benthic filter feeders / FCR3 : forage fish(미끼물고기 개체크기가 작은) / FCR4 : predatory fish(포식성 어류)
    #FCR4:메기, 쏘가리
    #FCR3:물고기
    #FCR2:물고기 무척추 동물로 선정
    #FCR2 : "11479" "11707" "11524" "11255"
    #FCR3 : "11193" "11420" "11052" "11289" "11252" "11423" "11348" "11690" "11265" "11414"
    #FCR4 : "11162" "11298"

    # 연령별로 노출량 계산
    intake_db_A_idx = intake_db[intake_db['age.x'] == 1].index.to_list()
    intake_db_B_idx = intake_db[(intake_db['age.x'] == 2) | (intake_db['age.x'] == 3)].index.to_list()
    intake_db_C_idx = intake_db[(intake_db['age.x'] >= 4) & (intake_db['age.x'] <= 6)].index.to_list()
    intake_db_D_idx = intake_db[(intake_db['age.x'] >= 7) & (intake_db['age.x'] <= 12)].index.to_list()
    intake_db_E_idx = intake_db[intake_db['age.x'] >= 13].index.to_list()

    def exp2_mean_value(x):
        Exp2 = Exp2_ir[x]
        Exp2_mean = Exp2.mean(axis=1)
        return Exp2_mean

    def exp3_mean_value(x):
        Exp3 = Exp3_ir[x]
        Exp3_mean = Exp3.mean(axis=1)
        return Exp3_mean

    def exp4_mean_value(x):
        Exp4 = Exp4_ir[x]
        Exp4_mean = Exp4.mean(axis=1)
        return Exp4_mean

    def expw_mean_value(x):
        Expw = Expw_ir[x]
        Expw_mean = Expw.mean(axis=1)
        return Expw_mean

    def exp_mean_value(x):
        Exp = Exp_ir[x]
        Exp_mean = Exp.mean(axis=1)
        return Exp_mean

    Exp2_A_mean=exp2_mean_value(intake_db_A_idx)
    Exp3_A_mean=exp3_mean_value(intake_db_A_idx)
    Exp4_A_mean=exp4_mean_value(intake_db_A_idx)
    Expw_A_mean=expw_mean_value(intake_db_A_idx)
    Exp_A_mean=exp_mean_value(intake_db_A_idx)

    Exp2_B_mean=exp2_mean_value(intake_db_B_idx)
    Exp3_B_mean=exp3_mean_value(intake_db_B_idx)
    Exp4_B_mean=exp4_mean_value(intake_db_B_idx)
    Expw_B_mean=expw_mean_value(intake_db_B_idx)
    Exp_B_mean=exp_mean_value(intake_db_B_idx)

    Exp2_C_mean=exp2_mean_value(intake_db_C_idx)
    Exp3_C_mean=exp3_mean_value(intake_db_C_idx)
    Exp4_C_mean=exp4_mean_value(intake_db_C_idx)
    Expw_C_mean=expw_mean_value(intake_db_C_idx)
    Exp_C_mean=exp_mean_value(intake_db_C_idx)

    Exp2_D_mean=exp2_mean_value(intake_db_D_idx)
    Exp3_D_mean=exp3_mean_value(intake_db_D_idx)
    Exp4_D_mean=exp4_mean_value(intake_db_D_idx)
    Expw_D_mean=expw_mean_value(intake_db_D_idx)
    Exp_D_mean=exp_mean_value(intake_db_D_idx)

    Exp2_E_mean=exp2_mean_value(intake_db_E_idx)
    Exp3_E_mean=exp3_mean_value(intake_db_E_idx)
    Exp4_E_mean=exp4_mean_value(intake_db_E_idx)
    Expw_E_mean=expw_mean_value(intake_db_E_idx)
    Exp_E_mean=exp_mean_value(intake_db_E_idx)

    Exp_mean = Exp_ir.mean(axis=1)
    return  material_conc,intake_db,intake_db_A_idx,intake_db_B_idx,intake_db_C_idx,intake_db_D_idx,intake_db_E_idx,T_if2,T_if3,T_if4,Exp_mean,\
            Exp2_A_mean,Exp3_A_mean,Exp4_A_mean,Expw_A_mean,Exp_A_mean,\
            Exp2_B_mean,Exp3_B_mean,Exp4_B_mean,Expw_B_mean,Exp_B_mean,\
            Exp2_C_mean,Exp3_C_mean,Exp4_C_mean,Expw_C_mean,Exp_C_mean,\
            Exp2_D_mean,Exp3_D_mean,Exp4_D_mean,Expw_D_mean,Exp_D_mean,\
            Exp2_E_mean,Exp3_E_mean,Exp4_E_mean,Expw_E_mean,Exp_E_mean