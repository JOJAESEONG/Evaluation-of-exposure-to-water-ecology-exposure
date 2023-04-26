import pickle
import numpy as np
import os
import geopandas as gpd # GeoPandas(지오판다스)
import pandas as pd
from copy import deepcopy
from matplotlib.pyplot import legend
import plotly.express as px
import plotly.graph_objects as go

path = os.getcwd()
os.getcwd()

class Rich():
    def __init__ (self, rich_did, lu_rch_did, ru_rch_did, stream_flux, c_area, geometry):
        self.rich_did = rich_did
        self.lu_rch_did = lu_rch_did
        self.ru_rch_did = ru_rch_did
        self.stream_flux = stream_flux               ## 단위 m3/s
        self.c_area = c_area
        self.geometry = geometry
        self.stream_flux_estimated = None
        self.lu_child = None
        self.ru_child = None
        self.stp_family = None
        self.stp_site = False

        self.parent = None

        self.velocity = None                         ## 단위 m/s
        self.velocity_estimated = None
        self.rch_len = None
        self.r_time = None                           ## 단위 hour

        self.PEC_est = None
        self.mass_g = None
    
    def take_lu_child(self,lu_rich_class,):
        self.lu_child = lu_rich_class
        
    def take_ru_child(self,ru_rich_class,):
        self.ru_child = ru_rich_class

    def estimate_flux(self, parents_true_flux, parents_true_carea):
        # ~np.isnan(self.stream_flux_estimated)
        if self.stream_flux is None:
            self.stream_flux_estimated = parents_true_flux *(self.c_area/parents_true_carea)

        elif self.stream_flux is not None:
            print('참값 있음')

def take_map_df(chemical_input,koc_value,stp_remove,water_value,df,Kdeg_water_per_hour):
    chemical=chemical_input
    koc=koc_value
    stp_removal =stp_remove  # 물질별로 계산해서 나오는 값 제거율 #STP 제거율
    water = water_value # 물질별로 나오는 값인것으로 알고있음, 배경농도 g/m3 # 2-2의 수계값
    df_e_t = df #인당 물질 배출량,  g/(d*명) 1인 1일 물질사용량의 총합


    line_gdf_rtime = pickle.load(open(os.path.join(path,"lind_gdf_rtime.p",), "rb"))
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2004170700', 'MB_NM'] = '내성천'
    stp_gdf_info = pickle.load(open(os.path.join(path,"stp_gdf_pop.p",), "rb"))
    stp_gdf_info.loc[:,'방류량'] = np.float64(stp_gdf_info.loc[:,'방류량'])

    data_rich = line_gdf_rtime.loc[:,['RCH_DID', 'LU_RCH_DID', 'RU_RCH_DID','CUM_AREA', 'geometry_x']].values
    rich_class_list = []
    for i in range(len(line_gdf_rtime)):
        i_info_rich = data_rich[i]
        rich_class_list.append(Rich(i_info_rich[0],i_info_rich[1],i_info_rich[2],None ,i_info_rich[3],i_info_rich[4]))

    line_gdf_rtime.loc[:,'rich_class'] = rich_class_list

    # 유량   est_flux
    line_gdf_rtime.loc[:,'est_flux'] = line_gdf_rtime.loc[:,'est_flux'] * 86400
    line_gdf_rtime.loc[:,'유량'] = line_gdf_rtime.loc[:,'유량'] * 86400
    for i in range(len(line_gdf_rtime)):
        try:
            line_gdf_rtime.loc[i,'rich_class'].stream_flux_estimated = line_gdf_rtime.loc[i,'est_flux'] 
        except:
            pass
        try:
            line_gdf_rtime.loc[i,'rich_class'].stream_flux = line_gdf_rtime.loc[i,'유량'] 
        except:
            pass

        try:
            line_gdf_rtime.loc[i,'rich_class'].r_time = line_gdf_rtime.loc[i,'r_time']
        except:
            pass
        try:
            line_gdf_rtime.loc[i,'rich_class'].parent = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == line_gdf_rtime.loc[i,'parent_rch_did'],'rich_class'].values[0]
        except:
            pass

    k = Kdeg_water_per_hour # 반응속도상수 반감기 역수 h-1 #K 계산 업데이트 대기중 

    koc = float(koc_value) # 사용자 입력값, L/kg #KOC 있고 text_input8
    Foc_susp = 0.1 # kg_oc*kg_solid-1
    SUSP_water = 4 # mg_solid*L_water-1
    constant_term = koc * Foc_susp * SUSP_water * (10**-6)   # 무차원 상수 


    removal_ratio = float(stp_removal)/100  # 물질별로 계산해서 나오는 값 제거율 #STP 제거율
    base_c = float(water) # 물질별로 나오는 값인것으로 알고있음, 배경농도 g/m3 # 2-2의 수계값
    e_p = float(df_e_t['1인 1일 물질사용량 (g/day · 명)'].sum()) # 0.000085 #인당 물질 배출량,  g/(d*명) 1인 1일 물질사용량의 총합


    # mass_g

    def cal_mass_down(stp_r, mass_g_out, k_constant):
        if stp_r.parent is not None:
            down_mass = mass_g_out * np.exp(-1*k_constant*stp_r.r_time) #구간 잔존률 * 위에서온 mass

            if stp_r.parent.mass_g is None:
                stp_r.parent.mass_g = down_mass  
            else:
                before_mass = stp_r.parent.mass_g
                stp_r.parent.mass_g = before_mass + down_mass

            cal_mass_down(stp_r.parent, down_mass, k_constant)
        

    # 우선 농도계산은 하지 않고 강줄기별 mass만 계산
    for i in range(len(stp_gdf_info)):
        i_info = stp_gdf_info.loc[i,['Effluent', '인구수', 'RCH_DID']]
        
        population = i_info.인구수
        stp_r = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == i_info.RCH_DID].rich_class.values[0]
        if stp_r.r_time is not None:
        

            mass_g = (e_p * population)  # g/day   _ 물질 배출량
            mass_g_out = mass_g*(1-removal_ratio)/(1 + constant_term)   # 처리효과 흡착??효과 고려한 물질 out 양


            # 총 물질량 합산 위에서 흘러온것 + stp에서 배출한 것
            try:
                mass_before = stp_r.mass_g
                stp_r.mass_g = mass_before + mass_g_out  # 앞단에서 모든 강줄기에 배경농도는 미리 더해놓음
            except:
                stp_r.mass_g = mass_g_out

            cal_mass_down(stp_r, mass_g_out, k)


    # 0.2보다 크면 방류량 추가 요청
    def parent_flux_est2(i_r,eff):
        
        if (i_r.parent is not None) and (i_r.parent.stream_flux_estimated is not None):
            before_stream = i_r.parent.stream_flux_estimated
            i_r.parent.stream_flux_estimated = before_stream + eff
            
            # if i_r.parent.parent is not None:
            parent_flux_est2(i_r.parent,eff)


    for i in range(len(stp_gdf_info)):
        eff_stp = stp_gdf_info.loc[i,'Effluent']
        rch_did = stp_gdf_info.loc[i,'RCH_DID']

        matched_line = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == rch_did]
        i_rich = matched_line.rich_class.values[0]
        if i_rich.stream_flux_estimated is not None:
            
            flux = i_rich.stream_flux_estimated
            if eff_stp / flux >= 0.2:
                
                i_rich.stream_flux_estimated = flux + eff_stp
                parent_flux_est2(i_rich,eff_stp)
            else:            
                i_rich = i_rich
                i_rich.stream_flux_estimated = flux



        elif i_rich.stream_flux is not None:
            
            flux = i_rich.stream_flux
            if eff_stp / flux >= 0.2:
                
                i_rich.stream_flux_estimated = flux + eff_stp
                parent_flux_est2(i_rich,eff_stp)
            else:            
                i_rich = i_rich
                i_rich.stream_flux_estimated = flux
        
    for i in range(len(line_gdf_rtime)):
        i_rich = line_gdf_rtime.loc[i,'rich_class']
        
        if np.isnan(i_rich.stream_flux_estimated):
            if ~np.isnan(line_gdf_rtime.loc[i,'유량']):
                i_rich.stream_flux_estimated = line_gdf_rtime.loc[i,'유량']

        

    for i in range(len(line_gdf_rtime)):
        line_gdf_rtime.loc[i,'est_flux'] = line_gdf_rtime.loc[i,'rich_class'].stream_flux_estimated

    # 합천창녕보 리치 중 한곳 바로 위 리치로 유량 교체 요청
    temp_est_flux = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014060100','est_flux'].values[0]
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014040400','est_flux'] = temp_est_flux
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014040400','rich_class'].values[0].stream_flux_estimated = temp_est_flux

    # 갑천 유량측정지 본류합류전까지 유량 대체값 조정 요청
    parent_est_flux = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '3009061401','est_flux'].values[0]
    parent = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '3009061401','rich_class'].values[0].parent
    while True:
        parent.stream_flux_estimated = parent_est_flux
        line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == parent.rich_did,'est_flux'] = parent_est_flux
        if parent.rich_did == '3010010100':
            break
        parent = parent.parent


    # 추정 유량을 통해 농도 환산
    for i in range(len(line_gdf_rtime)):
        i_r = line_gdf_rtime.loc[i,'rich_class']
        if i_r.mass_g is not None:
            
            pec_est = (i_r.mass_g / (line_gdf_rtime.loc[i,'est_flux'])) + base_c

            line_gdf_rtime.loc[i,'rich_class'].PEC_est = pec_est
            line_gdf_rtime.loc[i,'PEC_WS'] = pec_est

    # # 안동댐 수문 하류 농도 배출농도로 일괄처리 요청
    # temp_rch = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2001110900','rich_class'].values[0]
    
    # parent = temp_rch.parent
    # while True:
    #     if parent.rich_did == '2003040300':
    #         break
    #     parent.PEC_est = temp_rch.PEC_est
    #     line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == parent.rich_did, 'PEC_WS'] = temp_rch.PEC_est
    #     parent = parent.parent


        

    line_gdf_rtime.loc[:,'mass'] = [r.mass_g for r in line_gdf_rtime.rich_class.values]
    line_gdf_last = gpd.GeoDataFrame(line_gdf_rtime.loc[~line_gdf_rtime.r_time.isna(),['RCH_DID','MB_NM','RCH_LEN','geometry_x','PEC_WS','r_time','유량','est_flux']], geometry = 'geometry_x')
    line_gdf_last.columns = ['RCH_DID','MB_NM','RCH_LEN','geometry_x','PEC_WS','구간체류시간(h)','유량(m3/day)','추정유량(m3/day)']
    line_gdf_last = line_gdf_last.loc[:,['geometry_x','RCH_DID','MB_NM','RCH_LEN','유량(m3/day)','추정유량(m3/day)','구간체류시간(h)','PEC_WS',]]
    line_gdf_last.index = range(len(line_gdf_last))





    stp_gdf_info_2 = stp_gdf_info.loc[[b in line_gdf_last.RCH_DID.tolist() for b in stp_gdf_info.RCH_DID],['RCH_DID','시설명', '방류량', 'geometry', '인구수']]

    test = pd.merge(stp_gdf_info_2, line_gdf_last, how = 'left', on = 'RCH_DID')
    for i in range(len(test)):
        
        if ~ np.isnan(test.loc[i,'추정유량(m3/day)']):
            d_factor_i = (test.loc[i,'추정유량(m3/day)']) / (test.loc[i,'방류량'])
            if 0< d_factor_i < 1:
                # 희석배율 1미만인지역 1로 맞추고 변경된 추정유량으로 PEC_WS 변경
                test.loc[i,'희석배율'] = 1
                before_est_flux = test.loc[i,'추정유량(m3/day)']
                grad_flux = before_est_flux/test.loc[i,'방류량']
                test.loc[i,'추정유량(m3/day)'] = test.loc[i,'방류량']

                line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'추정유량(m3/day)'] = test.loc[i,'방류량']

                # line_gdf_rtime.loc[i,'PEC_WS']
                line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'] = (line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'] - base_c) * grad_flux + base_c
                test.loc[i,'PEC_WS'] = line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'].values[0]

            else:
                test.loc[i,'희석배율'] = d_factor_i

            PEC_local_i = (test.loc[i,'인구수'] * e_p * (1-removal_ratio)/(1 + constant_term)) / (test.loc[i,'추정유량(m3/day)'])


        test.loc[i,'C_local'] = (test.loc[i,'인구수'] * e_p * (1-removal_ratio)/(1 + constant_term)) / (test.loc[i,'방류량'])
        test.loc[i,'PEC_local'] = PEC_local_i
        


    remove_ind_list = []
    for i in range(len(line_gdf_last)):
        if line_gdf_last.loc[i,'RCH_DID'] not in set(test.RCH_DID):
            remove_ind_list.append(i)

    line_gdf_last_remove_dup = line_gdf_last.loc[remove_ind_list]
    line_gdf_last_remove_dup.index = range(len(line_gdf_last_remove_dup))

    total_df = pd.concat([test,line_gdf_last_remove_dup])
    total_df.index = range(len(total_df))

    test = test.loc[:,['RCH_DID','시설명','방류량','geometry', '인구수','희석배율','C_local','PEC_local',]]

    bpr_pec = ((10000 * e_p * (1-removal_ratio)/(1 + constant_term)) / 20000) + base_c
    # line_gdf_last = gpd.GeoDataFrame(pd.merge(line_gdf_last, line_gdf_rtime.loc[:,['RCH_DID', 'rich_class']], how = 'left', on='RCH_DID'), geometry = 'geometry_x')
    
    return line_gdf_last, test, total_df, bpr_pec,

def take_map_df_2(chemical_input,koc_value,stp_remove,water_value,df,Kdeg_water_per_hour):
    chemical=chemical_input
    koc=koc_value
    stp_removal =stp_remove  # 물질별로 계산해서 나오는 값 제거율 #STP 제거율
    water = water_value # 물질별로 나오는 값인것으로 알고있음, 배경농도 g/m3 # 2-2의 수계값
    df_e_t = df #인당 물질 배출량,  g/(d*명) 1인 1일 물질사용량의 총합

    line_gdf_rtime = pickle.load(open(os.path.join(path,"lind_gdf_rtime.p",), "rb"))
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2004170700', 'MB_NM'] = '내성천'
    stp_gdf_info = pickle.load(open(os.path.join(path,"stp_gdf_pop.p",), "rb"))
    stp_gdf_info.loc[:,'방류량'] = np.float64(stp_gdf_info.loc[:,'방류량'])

    data_rich = line_gdf_rtime.loc[:,['RCH_DID', 'LU_RCH_DID', 'RU_RCH_DID','CUM_AREA', 'geometry_x']].values
    rich_class_list = []
    for i in range(len(line_gdf_rtime)):
        i_info_rich = data_rich[i]
        rich_class_list.append(Rich(i_info_rich[0],i_info_rich[1],i_info_rich[2],None ,i_info_rich[3],i_info_rich[4]))

    line_gdf_rtime.loc[:,'rich_class'] = rich_class_list

    # 유량   est_flux
    line_gdf_rtime.loc[:,'est_flux'] = line_gdf_rtime.loc[:,'est_flux'] * 86400
    line_gdf_rtime.loc[:,'유량'] = line_gdf_rtime.loc[:,'유량'] * 86400
    for i in range(len(line_gdf_rtime)):
        try:
            line_gdf_rtime.loc[i,'rich_class'].stream_flux_estimated = line_gdf_rtime.loc[i,'est_flux'] 
        except:
            pass
        try:
            line_gdf_rtime.loc[i,'rich_class'].stream_flux = line_gdf_rtime.loc[i,'유량'] 
        except:
            pass

        try:
            line_gdf_rtime.loc[i,'rich_class'].r_time = line_gdf_rtime.loc[i,'r_time']
        except:
            pass
        try:
            line_gdf_rtime.loc[i,'rich_class'].parent = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == line_gdf_rtime.loc[i,'parent_rch_did'],'rich_class'].values[0]
        except:
            pass

    k = Kdeg_water_per_hour # 반응속도상수 반감기 역수 h-1

    koc = float(koc_value) # 사용자 입력값, L/kg #KOC 있고 text_input8
    Foc_susp = 0.1 # kg_oc*kg_solid-1
    SUSP_water = 4 # mg_solid*L_water-1
    constant_term = koc * Foc_susp * SUSP_water * (10**-6)   # 무차원 상수 


    removal_ratio = float(stp_removal)/100  # 물질별로 계산해서 나오는 값 제거율 #STP 제거율
    base_c = float(water) # 물질별로 나오는 값인것으로 알고있음, 배경농도 g/m3 # 2-2의 수계값
    e_p = float(df_e_t['1인 1일 물질사용량 (g/day · 명)'].sum()) # 0.000085 #인당 물질 배출량,  g/(d*명) 1인 1일 물질사용량의 총합

    # mass_g

    def cal_mass_down(stp_r, mass_g_out, k_constant):
        if stp_r.parent is not None:
            down_mass = mass_g_out * np.exp(-1*k_constant*stp_r.r_time) #구간 잔존률 * 위에서온 mass

            if stp_r.parent.mass_g is None:
                stp_r.parent.mass_g = down_mass  
            else:
                before_mass = stp_r.parent.mass_g
                stp_r.parent.mass_g = before_mass + down_mass

            cal_mass_down(stp_r.parent, down_mass, k_constant)
        

    # 우선 농도계산은 하지 않고 강줄기별 mass만 계산
    for i in range(len(stp_gdf_info)):
        i_info = stp_gdf_info.loc[i,['Effluent', '인구수', 'RCH_DID']]
        
        population = i_info.인구수
        stp_r = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == i_info.RCH_DID].rich_class.values[0]
        if stp_r.r_time is not None:
        

            mass_g = (e_p * population)  # g/day   _ 물질 배출량
            mass_g_out = mass_g*(1-removal_ratio)/(1 + constant_term)   # 처리효과 흡착??효과 고려한 물질 out 양


            # 총 물질량 합산 위에서 흘러온것 + stp에서 배출한 것
            try:
                mass_before = stp_r.mass_g
                stp_r.mass_g = mass_before + mass_g_out  # 앞단에서 모든 강줄기에 배경농도는 미리 더해놓음
            except:
                stp_r.mass_g = mass_g_out

            cal_mass_down(stp_r, mass_g_out, k)


    # 0.2보다 크면 방류량 추가 요청
    def parent_flux_est2(i_r,eff):
        
        if (i_r.parent is not None) and (i_r.parent.stream_flux_estimated is not None):
            before_stream = i_r.parent.stream_flux_estimated
            i_r.parent.stream_flux_estimated = before_stream + eff
            
            # if i_r.parent.parent is not None:
            parent_flux_est2(i_r.parent,eff)


    for i in range(len(stp_gdf_info)):
        eff_stp = stp_gdf_info.loc[i,'Effluent']
        rch_did = stp_gdf_info.loc[i,'RCH_DID']

        matched_line = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == rch_did]
        i_rich = matched_line.rich_class.values[0]
        if i_rich.stream_flux_estimated is not None:
            
            flux = i_rich.stream_flux_estimated
            if eff_stp / flux >= 0.2:
                
                i_rich.stream_flux_estimated = flux + eff_stp
                parent_flux_est2(i_rich,eff_stp)
            else:            
                i_rich = i_rich
                i_rich.stream_flux_estimated = flux



        elif i_rich.stream_flux is not None:
            
            flux = i_rich.stream_flux
            if eff_stp / flux >= 0.2:
                
                i_rich.stream_flux_estimated = flux + eff_stp
                parent_flux_est2(i_rich,eff_stp)
            else:            
                i_rich = i_rich
                i_rich.stream_flux_estimated = flux
        
    for i in range(len(line_gdf_rtime)):
        i_rich = line_gdf_rtime.loc[i,'rich_class']
        
        if np.isnan(i_rich.stream_flux_estimated):
            if ~np.isnan(line_gdf_rtime.loc[i,'유량']):
                i_rich.stream_flux_estimated = line_gdf_rtime.loc[i,'유량']

        

    for i in range(len(line_gdf_rtime)):
        line_gdf_rtime.loc[i,'est_flux'] = line_gdf_rtime.loc[i,'rich_class'].stream_flux_estimated

    # 합천창녕보 리치 중 한곳 바로 위 리치로 유량 교체 요청
    temp_est_flux = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014060100','est_flux'].values[0]
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014040400','est_flux'] = temp_est_flux
    line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2014040400','rich_class'].values[0].stream_flux_estimated = temp_est_flux

    # 갑천 유량측정지 본류합류전까지 유량 대체값 조정 요청
    parent_est_flux = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '3009061401','est_flux'].values[0]
    parent = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '3009061401','rich_class'].values[0].parent
    while True:
        parent.stream_flux_estimated = parent_est_flux
        line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == parent.rich_did,'est_flux'] = parent_est_flux
        if parent.rich_did == '3010010100':
            break
        parent = parent.parent


    # 추정 유량을 통해 농도 환산
    for i in range(len(line_gdf_rtime)):
        i_r = line_gdf_rtime.loc[i,'rich_class']
        if i_r.mass_g is not None:
            
            pec_est = (i_r.mass_g / (line_gdf_rtime.loc[i,'est_flux'])) + base_c

            line_gdf_rtime.loc[i,'rich_class'].PEC_est = pec_est
            line_gdf_rtime.loc[i,'PEC_WS'] = pec_est

    # # 안동댐 수문 하류 농도 배출농도로 일괄처리 요청
    # temp_rch = line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == '2001110900','rich_class'].values[0]
    
    # parent = temp_rch.parent
    # while True:
    #     if parent.rich_did == '2003040300':
    #         break
    #     parent.PEC_est = temp_rch.PEC_est
    #     line_gdf_rtime.loc[line_gdf_rtime.RCH_DID == parent.rich_did, 'PEC_WS'] = temp_rch.PEC_est
    #     parent = parent.parent


        

    line_gdf_rtime.loc[:,'mass'] = [r.mass_g for r in line_gdf_rtime.rich_class.values]
    line_gdf_last = gpd.GeoDataFrame(line_gdf_rtime.loc[~line_gdf_rtime.r_time.isna(),['RCH_DID','MB_NM','RCH_LEN','geometry_x','PEC_WS','r_time','유량','est_flux']], geometry = 'geometry_x')
    line_gdf_last.columns = ['RCH_DID','MB_NM','RCH_LEN','geometry_x','PEC_WS','구간체류시간(h)','유량(m3/day)','추정유량(m3/day)']
    line_gdf_last = line_gdf_last.loc[:,['geometry_x','RCH_DID','MB_NM','RCH_LEN','유량(m3/day)','추정유량(m3/day)','구간체류시간(h)','PEC_WS',]]
    line_gdf_last.index = range(len(line_gdf_last))





    stp_gdf_info_2 = stp_gdf_info.loc[[b in line_gdf_last.RCH_DID.tolist() for b in stp_gdf_info.RCH_DID],['RCH_DID','시설명', '방류량', 'geometry', '인구수']]

    test = pd.merge(stp_gdf_info_2, line_gdf_last, how = 'left', on = 'RCH_DID')
    for i in range(len(test)):
        
        if ~ np.isnan(test.loc[i,'추정유량(m3/day)']):
            d_factor_i = (test.loc[i,'추정유량(m3/day)']) / (test.loc[i,'방류량'])
            if 0< d_factor_i < 1:
                # 희석배율 1미만인지역 1로 맞추고 변경된 추정유량으로 PEC_WS 변경
                test.loc[i,'희석배율'] = 1
                before_est_flux = test.loc[i,'추정유량(m3/day)']
                grad_flux = before_est_flux/test.loc[i,'방류량']
                test.loc[i,'추정유량(m3/day)'] = test.loc[i,'방류량']

                line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'추정유량(m3/day)'] = test.loc[i,'방류량']

                # line_gdf_rtime.loc[i,'PEC_WS']
                line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'] = (line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'] - base_c) * grad_flux + base_c
                test.loc[i,'PEC_WS'] = line_gdf_last.loc[line_gdf_last.RCH_DID == test.loc[i,'RCH_DID'],'PEC_WS'].values[0]

            else:
                test.loc[i,'희석배율'] = d_factor_i

            PEC_local_i = (test.loc[i,'인구수'] * e_p * (1-removal_ratio)/(1 + constant_term)) / (test.loc[i,'추정유량(m3/day)'])


        test.loc[i,'C_local'] = (test.loc[i,'인구수'] * e_p * (1-removal_ratio)/(1 + constant_term)) / (test.loc[i,'방류량'])
        test.loc[i,'PEC_local'] = PEC_local_i
        


    remove_ind_list = []
    for i in range(len(line_gdf_last)):
        if line_gdf_last.loc[i,'RCH_DID'] not in set(test.RCH_DID):
            remove_ind_list.append(i)

    line_gdf_last_remove_dup = line_gdf_last.loc[remove_ind_list]
    line_gdf_last_remove_dup.index = range(len(line_gdf_last_remove_dup))

    total_df = pd.concat([test,line_gdf_last_remove_dup])
    total_df.index = range(len(total_df))

    test = test.loc[:,['RCH_DID','시설명','방류량','geometry', '인구수','희석배율','C_local','PEC_local',]]

    bpr_pec = ((10000 * e_p * (1-removal_ratio)/(1 + constant_term)) / 20000) + base_c
    line_gdf_last = gpd.GeoDataFrame(pd.merge(line_gdf_last, line_gdf_rtime.loc[:,['RCH_DID', 'rich_class']], how = 'left', on='RCH_DID'), geometry = 'geometry_x')
    
    return line_gdf_last, test

def get_stream_pec_plot(site: str,chemical_input,koc_value,stp_remove,water_value,df,Kdeg_water_per_hour):

    line_gdf_local, stp_point_gdf_local= take_map_df_2(chemical_input,koc_value,stp_remove,water_value,df,Kdeg_water_per_hour)

    stp_point_gdf_local = gpd.GeoDataFrame(pd.merge(line_gdf_local.loc[:,['RCH_DID', 'MB_NM']], stp_point_gdf_local, how = 'right', on='RCH_DID'),geometry = 'geometry')

    xx = site
    rch_list = []
    rch_len_list = []
    stp_list = []

    if xx != '전국':
        mb_gdf = line_gdf_local.loc[line_gdf_local.MB_NM == xx]
        # print(mb_gdf.columns.tolist())
    else:
        # print('전국 ㄴㄴ')
        return None

    for i in mb_gdf.index.to_list():
        ru_did = mb_gdf.rich_class[i].ru_rch_did
        lu_did = mb_gdf.rich_class[i].lu_rch_did
        
        #ru_rch_did, lu_rch_did, PEC_est
        

        if ru_did is None or ru_did not in mb_gdf.RCH_DID.tolist():
            if lu_did is None or lu_did not in mb_gdf.RCH_DID.tolist():
                rch_list_i = []
                rch_list_i.append(mb_gdf.rich_class[i].PEC_est)

                rch_len_list_i = []
                rch_len_list_i.append(line_gdf_local.loc[line_gdf_local.RCH_DID == mb_gdf.rich_class[i].rich_did].RCH_LEN.values[0])

                stp_list_i = []
                if len(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == mb_gdf.rich_class[i].rich_did, '시설명']) == 1:
                    stp_list_i.append(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == mb_gdf.rich_class[i].rich_did, '시설명'].values[0])
                else:
                    stp_list_i.append(None)

                ######## 이 뒤로 stp_list_i 붙이는 것 수정
                if mb_gdf.rich_class[i].parent is not None:
                    parent = mb_gdf.rich_class[i].parent
                    while True:
                        if parent.rich_did in mb_gdf.RCH_DID.tolist():
                            
                            rch_list_i.append(parent.PEC_est)
                            rch_len_list_i.append(line_gdf_local.loc[line_gdf_local.RCH_DID == parent.rich_did].RCH_LEN.values[0])

                            if len(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == parent.rich_did, '시설명']) == 1:
                                stp_list_i.append(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == parent.rich_did, '시설명'].values[0])
                            else:
                                stp_list_i.append(None)


                            if parent.parent is not None:
                                parent = parent.parent
                            else:
                                rch_len_list.append(rch_len_list_i)
                                rch_list.append(rch_list_i)
                                stp_list.append(stp_list_i)
                                break
                        else:
                            rch_len_list.append(rch_len_list_i)
                            rch_list.append(rch_list_i)
                            stp_list.append(stp_list_i)
                            break
                    

    for i in range(len(rch_len_list)):
        tem_list = deepcopy(rch_len_list[i])
        tem_list.reverse()
        tem_list = np.cumsum(tem_list).tolist()
        tem_list.reverse()
        rch_len_list[i] = tem_list

    df_list = []
    for i in range(len(rch_len_list)):
        t_df = pd.DataFrame(np.reshape(rch_list[i],(-1,1)), index=rch_len_list[i])
        df_list.append(t_df)

    df_list_stp = []
    for i in range(len(stp_list)):
        t_df = pd.DataFrame(np.reshape(stp_list[i],(-1,1)), index=rch_len_list[i])
        df_list_stp.append(t_df)
    stp_df = pd.concat(df_list_stp, axis=1).sort_index()
    stp_info_list_by_clen = []
    for i in range( len(stp_df)):
        stp_info_list_by_clen_i = []
        for i in list(set(stp_df.iloc[i].values)):
            if type(i) == str:
                stp_info_list_by_clen_i.append(i)
        if len(stp_info_list_by_clen_i) == 0:
            stp_info_list_by_clen_i.append(None)
        stp_info_list_by_clen.extend(stp_info_list_by_clen_i)
                
    stp_df.loc[:,'stp_name'] = stp_info_list_by_clen
    stp_df.loc[:,'len'] = stp_df.index.tolist()
    stp_df.index = range(len(stp_df))
    stp_df = stp_df.loc[:,['stp_name', 'len']]


    df_list_2 = []
    for i in range(len(df_list)):
        t_df = deepcopy(df_list[i])
        t_df.loc[:,'len'] = t_df.index.tolist()
        t_df.rename(columns = {0:'PEC_ws',},inplace=True)
        t_df.index = range(len(t_df))
        t_df.loc[:,'rch'] = i
        df_list_2.append(t_df)

    t_df3 = pd.concat(df_list_2, axis=0)

    df_by_lenindex = pd.DataFrame(list(set(t_df3.len)), columns=['len']) # 듬성듬성하지만 빠름
    for i in range(len(df_list_2)):
        t_df = deepcopy(df_list_2[i])
        filled_df = pd.merge(df_by_lenindex, t_df, how='left', on = 'len').sort_values('len').fillna(method='backfill')
        df_list_2[i] = filled_df.dropna()

    filled_pec_line_df = pd.concat(df_list_2, axis=0)
    stp_df = pd.merge(stp_df.dropna(),t_df3.loc[:,['PEC_ws', 'len']], how='left', on = 'len').drop_duplicates()
    stp_df.index = range(len(stp_df))

    color_discrete_map = {}
    for i in list(set(filled_pec_line_df.rch)):
        color_discrete_map[i] = 'rgb(0,0,150)'



    fig = px.line(filled_pec_line_df, 
                x='len', # x축
                y='PEC_ws',  # y축
                color='rch',
                title=f'{xx}', # Title 
                color_discrete_map=color_discrete_map,
                )
    fig.update_layout(width=1350, height=450,showlegend=False)
    fig.update_xaxes(title="거리")
    fig.update_yaxes(title="PEC_WS(㎎/L)")
    # fig.update_layout(showlegend=False)

    color_list = [
            'rgba(255, 51, 51, .9)',
            'rgba(255, 122, 51, .9)',
            'rgba(255, 221, 51, .9)',
            'rgba(168, 255, 51, .9)',
            'rgba(51, 255, 209, .9)',
            'rgba(51, 110, 255, .9)',
            'rgba(212, 51, 255, .9)',
            'rgba(255, 51, 131, .9)',
            'rgba(183, 142, 158, .9)',
            'rgba(131, 45, 79, .9)',
            'rgba(49,   182, 100, .9)',
            'rgba(180, 80, 40, .9)',
            'rgba(30, 123, 97, .9)',
            'rgba(190, 51, 233, .9)',
            'rgba(240, 35, 211, .9)',
            'rgba(100, 87, 193, .9)',
            'rgba(30, 179, 142, .9)',
            'rgba(130, 98, 111, .9)',
            'rgba(170, 12, 193, .9)',
    ]

    for i in range(len(stp_df)):
        i_color = color_list[i]
        stp_point = go.Scatter(x=stp_df.loc[i,['len']],
                                    y=stp_df.loc[i,['PEC_ws']],
                                    mode="markers",
                                    # line=go.scatter.Line(color="gray"),
                                    marker={'size':10},
                                    marker_color=i_color,
                                    legendgroup="STP point",
                                    legendgrouptitle_text="STP point",
                                    name=stp_df.loc[i,['stp_name']].values[0],
                                    showlegend=True
                                    )
        # print(stp_df.loc[i,['stp_name']].values[0])

        fig.add_trace(stp_point)
    return fig

