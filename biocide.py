import numpy as np
import pandas as pd
import random
import math
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib import style
from bokeh.models import FuncTickFormatter

def user_input(input_chemical, M_input, P_vap_input, inh_abs_input, der_abs_input):
  

    product_content = pd.read_csv('product_chem_final.csv', encoding='CP949')  # 제품 DB
    alg_list = pd.read_csv('algo_final_ver.2(final).csv', encoding='CP949')  # 알고리즘 DB


    M = M_input
    P_vap = P_vap_input
    inh_abs = inh_abs_input
    der_abs = der_abs_input

    if 10.0 <= P_vap:
        F = 1
    if 1.0 <= P_vap < 10.0:
        F = 0.1
    if 0.1 <= P_vap < 1.0:
        F = 0.01
    if P_vap < 0.1:
        F = 0.001

    # 일시적 분사
    gen_m_alg1 = pd.read_csv('gen_m_alg1_exd.csv')
    gen_w_alg1 = pd.read_csv('gen_w_alg1_exd.csv')

    # 지속적 방출
    gen_m_alg2 = pd.read_csv('gen_m_alg2_exd.csv')
    gen_w_alg2 = pd.read_csv('gen_w_alg2_exd.csv')

    # 지속적 휘발
    gen_m_alg3 = pd.read_csv('gen_m_alg3_exd.csv')
    gen_w_alg3 = pd.read_csv('gen_w_alg3_exd.csv')

    # 간헐적 휘발
    gen_m_alg4 = pd.read_csv('gen_m_alg4_exd.csv')
    gen_w_alg4 = pd.read_csv('gen_w_alg4_exd.csv')

    # 피부접촉 - 사용량 기반
    gen_m_alg5 = pd.read_csv('gen_m_alg5_exd.csv')
    gen_w_alg5 = pd.read_csv('gen_w_alg5_exd.csv')

    # 피부접촉 - 사용시간 기반
    gen_m_alg6 = pd.read_csv('gen_m_alg6_exd.csv')
    gen_w_alg6 = pd.read_csv('gen_w_alg6_exd.csv')

    selected_product_content = product_content[product_content["CAS"] == input_chemical]
    selected_product_list = list(set(selected_product_content["code"]))

# 살생물제품 코드는 제외하기
    # except_list = []
    # for i in range(len(selected_product_list)):
    #     if selected_product_list[i].startswith("L"):
    #         except_list.append(selected_product_list[i])
    #     elif selected_product_list[i].startswith("K"):
    #         except_list.append(selected_product_list[i])
    #     elif selected_product_list[i].startswith("J"):
    #         except_list.append(selected_product_list[i])
    #     elif selected_product_list[i] == "M1_6":
    #         except_list.append(selected_product_list[i])

    # selected_product_list = list(set(selected_product_list) - set(except_list))
    # del_code = set(selected_product_list) - set(gen_m_alg1.columns) # 이용행태 데이터에 없는 제품코드는 제거
    # selected_product_list = list(set(selected_product_list) - del_code)
    selected_product_list.sort() # 리스트 정렬(순서 고정)

    selected_alg_list = alg_list[alg_list["code"].isin(selected_product_list)]



##############################################################################

    selected_gen_m_alg1 = gen_m_alg1.loc[:, selected_product_list]
    selected_gen_w_alg1 = gen_w_alg1.loc[:, selected_product_list]

    selected_gen_m_alg2 = gen_m_alg2.loc[:, selected_product_list]
    selected_gen_w_alg2 = gen_w_alg2.loc[:, selected_product_list]

    selected_gen_m_alg3 = gen_m_alg3.loc[:, selected_product_list]
    selected_gen_w_alg3 = gen_w_alg3.loc[:, selected_product_list]

    selected_gen_m_alg4 = gen_m_alg4.loc[:, selected_product_list]
    selected_gen_w_alg4 = gen_w_alg4.loc[:, selected_product_list]

    selected_gen_m_alg5 = gen_m_alg5.loc[:, selected_product_list]
    selected_gen_w_alg5 = gen_w_alg5.loc[:, selected_product_list]

    selected_gen_m_alg6 = gen_m_alg6.loc[:, selected_product_list]
    selected_gen_w_alg6 = gen_w_alg6.loc[:, selected_product_list]

##############################################################################

    use_rate_df = pd.read_csv('use_rate_200819.csv', encoding='CP949')

    m_use_rate_df = use_rate_df[use_rate_df['SQ1'] == '남자']
    w_use_rate_df = use_rate_df[use_rate_df['SQ1'] == '여자']

    selected_m_use_rate = m_use_rate_df[selected_product_list]
    selected_w_use_rate = w_use_rate_df[selected_product_list]


    selected_use_rate = pd.concat([selected_m_use_rate, selected_w_use_rate])

    selected_gen_m_alg1 = pd.DataFrame(np.array(selected_gen_m_alg1) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)
    selected_gen_m_alg2 = pd.DataFrame(np.array(selected_gen_m_alg2) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)
    selected_gen_m_alg3 = pd.DataFrame(np.array(selected_gen_m_alg3) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)
    selected_gen_m_alg4 = pd.DataFrame(np.array(selected_gen_m_alg4) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)
    selected_gen_m_alg5 = pd.DataFrame(np.array(selected_gen_m_alg5) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)
    selected_gen_m_alg6 = pd.DataFrame(np.array(selected_gen_m_alg6) *
                                       np.array(selected_m_use_rate), columns=selected_product_list)

    selected_gen_w_alg1 = pd.DataFrame(np.array(selected_gen_w_alg1) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)
    selected_gen_w_alg2 = pd.DataFrame(np.array(selected_gen_w_alg2) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)
    selected_gen_w_alg3 = pd.DataFrame(np.array(selected_gen_w_alg3) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)
    selected_gen_w_alg4 = pd.DataFrame(np.array(selected_gen_w_alg4) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)
    selected_gen_w_alg5 = pd.DataFrame(np.array(selected_gen_w_alg5) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)
    selected_gen_w_alg6 = pd.DataFrame(np.array(selected_gen_w_alg6) *
                                       np.array(selected_w_use_rate), columns=selected_product_list)

    m_alg1_ei = np.array(selected_gen_m_alg1)
    m_alg2_ei = np.array(selected_gen_m_alg2)
    m_alg3_ei = np.array(selected_gen_m_alg3)
    m_alg4_ei = np.array(selected_gen_m_alg4)
    m_alg5_ei = np.array(selected_gen_m_alg5)
    m_alg6_ei = np.array(selected_gen_m_alg6)

    m_ei_array = np.stack((m_alg1_ei, m_alg2_ei, m_alg3_ei, m_alg4_ei, m_alg5_ei, m_alg6_ei))

    w_alg1_ei = np.array(selected_gen_w_alg1)
    w_alg2_ei = np.array(selected_gen_w_alg2)
    w_alg3_ei = np.array(selected_gen_w_alg3)
    w_alg4_ei = np.array(selected_gen_w_alg4)
    w_alg5_ei = np.array(selected_gen_w_alg5)
    w_alg6_ei = np.array(selected_gen_w_alg6)

    w_ei_array = np.stack((w_alg1_ei, w_alg2_ei, w_alg3_ei, w_alg4_ei, w_alg5_ei, w_alg6_ei))
    
    alg_name = {"airborne_short": "일시적 분사", "airborne_release": "지속적 방출", "conti_release": "지속적 휘발",
                "surface_volatilization": "간헐적 휘발", "liquid_contact": "피부접촉(사용량 기반)",
                "spraying_contact": "피부접촉(사용시간 기반)"}

    gen_alg1 = pd.concat([gen_m_alg1, gen_w_alg1])
    gen_alg2 = pd.concat([gen_m_alg2, gen_w_alg2])
    gen_alg3 = pd.concat([gen_m_alg3, gen_w_alg3])
    gen_alg4 = pd.concat([gen_m_alg4, gen_w_alg4])
    gen_alg5 = pd.concat([gen_m_alg5, gen_w_alg5])
    gen_alg6 = pd.concat([gen_m_alg6, gen_w_alg6])

######################################################
    def W_f_matrix(prod_list, prod_content):
        m = []
        for i in range(len(prod_list)):  # i : 물질이 포함된 제품
            content_i = prod_content[prod_content["code"] == prod_list[i]]
            mean_conc_max = np.array(content_i.loc[:, "conc_max"])
            mean_conc_max = mean_conc_max.astype('float')
            W_f = np.mean(mean_conc_max) / 100  # ************************************수정된 부분***************************
            m.append(W_f)
        m = np.array(m)
        m = m.T
        return m

    selected_W_f = W_f_matrix(selected_product_list, product_content)

    def F_air_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            F_air_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            F_air_value = np.array(F_air_i.loc[:, "F_air"])
            F_air_value = F_air_value.astype('float')
            F_air_value = np.mean(F_air_value)
            m.append(F_air_value)
        m = np.array(m)
        m = m.T
        return m

    selected_F_air = F_air_matrix(selected_product_list, selected_alg_list)

    def R_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            R_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            R_value = np.array(R_i.loc[:, "R"])
            R_value = R_value.astype('float')
            R_value = np.mean(R_value)
            m.append(R_value)
        m = np.array(m)
        m = m.T
        return m

    selected_R = R_matrix(selected_product_list, selected_alg_list)

    def N_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            N_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            N_value = np.array(N_i.loc[:, "N"])
            N_value = N_value.astype('float')
            N_value = np.mean(N_value)
            m.append(N_value)
        m = np.array(m)
        m = m.T
        return m

    selected_N = N_matrix(selected_product_list, selected_alg_list)

    def V_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            V_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            V_value = np.array(V_i.loc[:, "V"])
            V_value = V_value.astype('float')
            V_value = np.mean(V_value)
            m.append(V_value)
        m = np.array(m)
        m = m.T
        return m

    selected_V = V_matrix(selected_product_list, selected_alg_list)

    def S_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            S_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            S_value = np.array(S_i.loc[:, "S"])
            S_value = S_value.astype('float')
            S_value = np.mean(S_value)
            m.append(S_value)
        m = np.array(m)
        m = m.T
        return m

    selected_S = S_matrix(selected_product_list, selected_alg_list)

    def M_r_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            M_r_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            M_r_value = np.array(M_r_i.loc[:, "Mr"])
            M_r_value = M_r_value.astype('float')
            M_r_value = np.mean(M_r_value)
            m.append(M_r_value)
        m = np.array(m)
        m = m.T
        return m

    selected_M_r = M_r_matrix(selected_product_list, selected_alg_list)

    def t_matrix(prod_list, algorithm_list):
        m = []
        for i in range(len(prod_list)):
            t_i = algorithm_list[algorithm_list["code"] == prod_list[i]]
            t_value = np.array(t_i.loc[:, "t"])
            t_value = t_value.astype('float')
            t_value = np.mean(t_value)
            m.append(t_value)
        m = np.array(m)
        m = m.T
        return m

    selected_t = t_matrix(selected_product_list, selected_alg_list)

    R_c = 8.3144  # 보편기체상수 [mg*m^2/min^2*mole*K]
    T = 293.15  # [K]
    K = np.sqrt(R_c * T / (2 * 3.1415 * M))
    IR = 1.2 / 60  # 호흡률 [m3/min]
    mass = 0.0097  # *************************추가된 부분**********************
    V_prod = 0.00000000978  # *************************추가된 부분**********************
    D = mass / V_prod  # *************************추가된 부분**********************

    def evaporation(x, t):  # *************************추가된 부분~~~~~~~~~
        C_air = x[0]
        C_v = x[1]
        C_air_bar = x[2]
        time_averaged_C_air = x[3]
        dC_airdt = K * S * M * P_vap * C_v / ((C_v + (D - C_v) * M / M_r) * V_room * R_c * T) - (
                    K * S / V_room + Q) * C_air
        dC_vdt = -K * S * M * (P_vap * C_v / (C_v + (D - C_v) * M / M_r)) / (R_c * T * V_prod) + K * S * C_air / V_prod
        dC_air_bardt = C_air
        dtime_averaged_C_airdt = -C_air_bar / (t + 0.0001) ** 2 + C_air / (t + 0.0001)
        return [dC_airdt, dC_vdt, dC_air_bardt, dtime_averaged_C_airdt]

    TWA = []

    for i in range(len(selected_product_list)):
        if selected_alg_list['conti_release'][selected_alg_list['code'] == selected_product_list[i]].values[0] == 0:
            TWA.append(0)
        else:
            S = selected_S[i]
            Q = selected_N[i] / 60
            V_room = selected_V[i]
            M_r = selected_M_r[i]
            C_air_ini = 0.0
            C_v_ini = D * selected_W_f[i]
            C_air_bar_ini = 0.0
            time_averaged_C_air_ini = 0.0
            time = np.linspace(0, np.int64(selected_t[i]), np.int64(selected_t[i]) + 1)
            y = odeint(evaporation, [C_air_ini, C_v_ini, C_air_bar_ini, time_averaged_C_air_ini], time)
            time_averaged_C_air = y[:, 3]
            TWA.append(time_averaged_C_air[np.int64(selected_t[i])])
    TWA = np.array(TWA)  # ~~~~~~~~~~추가된 부분**********************

    def func_array(N, V, t, S, M_r, W_f, F_air, R):
        alg1_func = W_f * F_air / V * (1 - np.exp(-N * t)) / N * IR * inh_abs
        alg2_func = W_f * IR * inh_abs * t * (1 - ((1 - np.exp(-N * t)) / (N * t))) / (N * V)
        alg3_func = TWA * IR * inh_abs * t  # ********************수정된 부분***********************************
        alg4_func = W_f * F * ((1 - np.exp(-N * t)) / N) * IR * inh_abs / V
        alg5_func = W_f * 0.01 * der_abs
        alg6_func = R * W_f * der_abs
        func_array = np.stack((alg1_func, alg2_func, alg3_func, alg4_func, alg5_func, alg6_func))
        return func_array

    # m_alg1_ei = m_alg1_ei.astype('int')
    # w_alg1_ei = w_alg1_ei.astype('int')
    m_func_array = func_array(selected_N, selected_V, selected_t, selected_S, selected_M_r, selected_W_f,
                              selected_F_air, selected_R)
    w_func_array = func_array(selected_N, selected_V, selected_t, selected_S, selected_M_r, selected_W_f,
                              selected_F_air, selected_R)

    m_func_array[np.isnan(m_func_array)] = 0
    w_func_array[np.isnan(w_func_array)] = 0

    m_ei_array[pd.isnull(m_ei_array)] = 0  # 추가된 부분 - 200820
    w_ei_array[pd.isnull(w_ei_array)] = 0

    m_func_array1 = np.tensordot(np.ones(3124), m_func_array, 0)
    w_func_array1 = np.tensordot(np.ones(2991), w_func_array, 0)

    m_func_array = m_func_array1.reshape((6, 3124, len(selected_product_list)))
    w_func_array = w_func_array1.reshape((6, 2991, len(selected_product_list)))

    # 개인별 제품별 알고리즘별 노출량 배열
    m_array = m_ei_array * m_func_array

    w_array = w_ei_array * w_func_array

    # 제품별 노출알고리즘 반영
    selected_alg_list = selected_alg_list.sort_values(by="code")

    indicator_matrix = selected_alg_list[
        ["airborne_short", "airborne_release", "conti_release", "surface_volatilization",
         "liquid_contact", "spraying_contact"]]
    indicator_matrix = np.array(indicator_matrix)
    indicator_matrix = indicator_matrix.T  # 행 : 알고리즘, 열 : 제품

    m_indicator_array = np.tensordot(np.ones(3124), indicator_matrix, 0)  # (개인, 알고리즘, 제품)
    w_indicator_array = np.tensordot(np.ones(2991), indicator_matrix, 0)  # (개인, 알고리즘, 제품)

    m_indicator_array_T = np.transpose(m_indicator_array, [1, 0, 2])  # (알고리즘,개인,제품)


    w_indicator_array_T = np.transpose(w_indicator_array, [1, 0, 2])  # (알고리즘,개인,제품)

    # 개인별 제품별 알고리즘별 노출량 배열

    m_exposure_array = m_array * m_indicator_array_T  # (알고리즘, 개인, 제품)
    w_exposure_array = w_array * w_indicator_array_T  # (알고리즘, 개인, 제품)

    # 개인별 제품별 노출량

    m_alg_sum = np.sum(m_exposure_array, axis=0)
    w_alg_sum = np.sum(w_exposure_array, axis=0)  # (개인, 제품)
    alg_sum = np.concatenate((m_alg_sum, w_alg_sum), axis=0)

    # 제품별 노출량 총합 구하기
    tot_prod = np.sum(alg_sum, axis=0)

    tot_prod_df = pd.DataFrame(tot_prod, columns=["total_exposure"])

    sort_tot_prod_df = tot_prod_df.sort_values(by=['total_exposure'], axis=0, ascending=False)

    # 개인별(남성) 사용제품 노출량 누적
    m_exposure = np.sum(m_alg_sum, axis=1)
    m_exposure = m_exposure.astype('float')
    m_exposure_df = pd.DataFrame(m_exposure)
    m_exposure_df.columns = ['exposure']
    m_exposure_df = m_exposure_df.round(3)

    # 개인별(여성) 사용제품 노출량 누적
    w_exposure = np.sum(w_alg_sum, axis=1)
    w_exposure = w_exposure.astype('float')
    w_exposure_df = pd.DataFrame(w_exposure)
    w_exposure_df.columns = ['exposure']
    w_exposure_df = w_exposure_df.round(3)

    # 개인별(전체) 사용제품 노출량 누적
    exposure = np.concatenate([m_exposure, w_exposure])
    exposure_df = pd.DataFrame(exposure)
    exposure_df.columns = ['exposure']
    exposure_df = exposure_df.round(3)

    # 흡입 노출량 계산
    # 200819 추가
    m_inh_alg_sum = np.sum(m_exposure_array[:4, :, :], axis=0)
    w_inh_alg_sum = np.sum(w_exposure_array[:4, :, :], axis=0)
    m_der_alg_sum = np.sum(m_exposure_array[4:, :, :], axis=0)
    w_der_alg_sum = np.sum(w_exposure_array[4:, :, :], axis=0)

    m_inh_exposure = np.sum(m_inh_alg_sum, axis=1)
    m_der_exposure = np.sum(m_der_alg_sum, axis=1)
    w_inh_exposure = np.sum(w_inh_alg_sum, axis=1)
    w_der_exposure = np.sum(w_der_alg_sum, axis=1)

    inh_exposure = np.concatenate([m_inh_exposure, w_inh_exposure])
    der_exposure = np.concatenate([m_der_exposure, w_der_exposure])

    result_df = pd.DataFrame(index=range(0, 223), columns=['CAS', 'inh_exposure', 'der_exposure'])


    # 한글 폰트 불러오기
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    import matplotlib

    # CDF
    import bokeh.plotting
    x1 = np.log10(np.sort(exposure))
    y1 = np.arange(1, len(x1) + 1) / len(x1)
    x2 = np.log10(np.sort(m_exposure))
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    x3 = np.log10(np.sort(w_exposure))
    y3 = np.arange(1, len(x3) + 1) / len(x3)
    p1 = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=600)
    p1.xaxis.axis_label = '노출량(mg/kg/day)'
    p1.yaxis.axis_label = '누적빈도'
    p1.title.text = '사용자 인구집단 누적 전신 노출량 분포'
    p1.line(x1, y1, line_color="#4673eb", line_width=4, alpha=0.3, legend_label="전체", color='#E69F00')
    p1.line(x2, y2, line_color="red", line_width=4, alpha=0.3, legend_label="남성", color='red')
    p1.line(x3, y3, line_color="green", line_width=4, alpha=0.3, legend_label="여성", color='green')
    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"
    p1.xaxis[0].formatter = FuncTickFormatter(code="""
      var str = tick.toString(); //get exponent
      var newStr = "";
      for (var i=0; i<10;i++)
      {
        var code = str.charCodeAt(i);
        switch(code) {
          case 45: // "-"
            newStr += "⁻";
            break;
          case 48: // "0"
            newStr +="⁰";
            break;
          case 49: // "1"
            newStr +="¹";
            break;
          case 50: // "2"
            newStr +="²";
            break;
          case 51: // "3"
            newStr +="³"
            break;
          case 52: // "4"
            newStr +="⁴"
            break;
          case 53: // "5"
            newStr +="⁵"
            break;                
          case 54: // "6"
            newStr +="⁶"
            break;
          case 55: // "7"
            newStr +="⁷"
            break;
          case 56: // "8"
            newStr +="⁸"
            break;
          case 57: // "9"
            newStr +="⁹"
            break;                         
        }
      }
      return 10+newStr;
    """)
    p1.axis.major_label_text_font = "sans"
###################################################################################################
    import bokeh.plotting
    x1 = np.log10(np.sort(inh_exposure))
    y1 = np.arange(1, len(x1) + 1) / len(x1)
    x2 = np.log10(np.sort(m_inh_exposure))
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    x3 = np.log10(np.sort(w_inh_exposure))
    y3 = np.arange(1, len(x3) + 1) / len(x3)
    p7 = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=600)
    p7.xaxis.axis_label = '노출량(mg/kg/day)'
    p7.yaxis.axis_label = '누적빈도'
    p7.title.text = '사용자 인구집단 누적 흡입 노출량 분포'
    p7.line(x1, y1, line_color="#4673eb", line_width=4, alpha=0.3, legend_label="전체", color='#E69F00')
    p7.line(x2, y2, line_color="red", line_width=4, alpha=0.3, legend_label="남성", color='red')
    p7.line(x3, y3, line_color="green", line_width=4, alpha=0.3, legend_label="여성", color='green')
    p7.legend.location = "top_left"
    p7.legend.click_policy="hide"
    p7.xaxis[0].formatter = FuncTickFormatter(code="""
      var str = tick.toString(); //get exponent
      var newStr = "";
      for (var i=0; i<10;i++)
      {
        var code = str.charCodeAt(i);
        switch(code) {
          case 45: // "-"
            newStr += "⁻";
            break;
          case 48: // "0"
            newStr +="⁰";
            break;
          case 49: // "1"
            newStr +="¹";
            break;
          case 50: // "2"
            newStr +="²";
            break;
          case 51: // "3"
            newStr +="³"
            break;
          case 52: // "4"
            newStr +="⁴"
            break;
          case 53: // "5"
            newStr +="⁵"
            break;                
          case 54: // "6"
            newStr +="⁶"
            break;
          case 55: // "7"
            newStr +="⁷"
            break;
          case 56: // "8"
            newStr +="⁸"
            break;
          case 57: // "9"
            newStr +="⁹"
            break;                         
        }
      }
      return 10+newStr;
    """)
    p7.axis.major_label_text_font = "sans"


###################################################################################################

    #bokeh.plotting.show(p1)
    # print(np.log10(np.percentile(exposure[exposure > 0], 95)))
    start_date = np.log10(np.percentile(exposure[exposure > 0], 95))
    ax1 = np.log10(exposure[exposure > 0])  # table
    ax1 = np.sort(ax1)
    p2 = bokeh.plotting.figure(sizing_mode="stretch_width", width=580, height=390)
    p2.xaxis.axis_label = '노출량(mg/kg/day)'
    p2.yaxis.axis_label = '빈도(상대빈도)'
    p2.title.text = '사용자 전신 노출량 분포' + '(n=' + str(round(np.sum(np.array(selected_use_rate).sum(axis=1) > 0))) + ')'

    p2_table_data = [format(np.percentile(exposure[exposure > 0], 5),'.2E'), format(np.percentile(exposure[exposure > 0], 50),'.2E'),
                     format(np.percentile(exposure[exposure > 0], 75),'.2E'), format(np.percentile(exposure[exposure > 0], 90),'.2E'),
                     format(np.percentile(exposure[exposure > 0], 95),'.2E'), format(np.percentile(exposure[exposure > 0], 99),'.2E')]

    hist, edges = np.histogram(ax1, density=True, bins=50)
    daylight_savings_start = bokeh.models.Span(location=start_date,
                                            dimension='height', line_color='red',
                                            line_dash='dashed', line_width=1)
    daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                    x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
    p2.renderers.extend([daylight_savings_start, daylight_savings_start_label])
    p2.quad(top=hist,bottom=0, color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])

    p2.xaxis[0].formatter = FuncTickFormatter(code="""
      var str = tick.toString(); //get exponent
      var newStr = "";
      for (var i=0; i<10;i++)
      {
        var code = str.charCodeAt(i);
        switch(code) {
          case 45: // "-"
            newStr += "⁻";
            break;
          case 48: // "0"
            newStr +="⁰";
            break;
          case 49: // "1"
            newStr +="¹";
            break;
          case 50: // "2"
            newStr +="²";
            break;
          case 51: // "3"
            newStr +="³"
            break;
          case 52: // "4"
            newStr +="⁴"
            break;
          case 53: // "5"
            newStr +="⁵"
            break;                
          case 54: // "6"
            newStr +="⁶"
            break;
          case 55: // "7"
            newStr +="⁷"
            break;
          case 56: // "8"
            newStr +="⁸"
            break;
          case 57: // "9"
            newStr +="⁹"
            break;                         
        }
      }
      return 10+newStr;
    """)
    p2.axis.major_label_text_font = "sans"

    start_date = np.log10(np.percentile(m_exposure[m_exposure > 0], 95))
    ax1 = np.log10(m_exposure[m_exposure > 0])  # table
    ax1 = np.sort(ax1)
    p3 = bokeh.plotting.figure(sizing_mode="stretch_width", width=580, height=390)
    p3.xaxis.axis_label = '노출량(mg/kg/day)'
    p3.yaxis.axis_label = '빈도(상대빈도)'
    p3.title.text = '남성 사용자 노출량 분포' + '(n=' + str(round(np.sum(selected_m_use_rate.sum(axis=1) > 0))) + ')'

    p3_table_data = [format(np.percentile(m_exposure[m_exposure > 0], 5),'.2E'), format(np.percentile(m_exposure[m_exposure > 0], 50),'.2E'),
                     format(np.percentile(m_exposure[m_exposure > 0], 75),'.2E'), format(np.percentile(m_exposure[m_exposure > 0], 90),'.2E'),
                     format(np.percentile(m_exposure[m_exposure > 0], 95),'.2E'), format(np.percentile(m_exposure[m_exposure > 0], 99),'.2E')]

    hist, edges = np.histogram(ax1, density=True, bins=50)
    daylight_savings_start = bokeh.models.Span(location=start_date,
                                        dimension='height', line_color='red',
                                        line_dash='dashed', line_width=1)
    daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                    x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
    p3.renderers.extend([daylight_savings_start, daylight_savings_start_label])
    p3.quad(top=hist,bottom=0, color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])

    p3.xaxis[0].formatter = FuncTickFormatter(code="""
      var str = tick.toString(); //get exponent
      var newStr = "";
      for (var i=0; i<10;i++)
      {
        var code = str.charCodeAt(i);
        switch(code) {
          case 45: // "-"
            newStr += "⁻";
            break;
          case 48: // "0"
            newStr +="⁰";
            break;
          case 49: // "1"
            newStr +="¹";
            break;
          case 50: // "2"
            newStr +="²";
            break;
          case 51: // "3"
            newStr +="³"
            break;
          case 52: // "4"
            newStr +="⁴"
            break;
          case 53: // "5"
            newStr +="⁵"
            break;                
          case 54: // "6"
            newStr +="⁶"
            break;
          case 55: // "7"
            newStr +="⁷"
            break;
          case 56: // "8"
            newStr +="⁸"
            break;
          case 57: // "9"
            newStr +="⁹"
            break;                         
        }
      }
      return 10+newStr;
    """)
    p3.axis.major_label_text_font = "sans"


    start_date = np.log10(np.percentile(w_exposure[w_exposure > 0], 95))
    ax1 = np.log10(w_exposure[w_exposure > 0])
    ax1 = np.sort(ax1)
    p4 = bokeh.plotting.figure(sizing_mode="stretch_width", width=580, height=390)
    p4.xaxis.axis_label = '노출량(mg/kg/day)'
    p4.yaxis.axis_label = '빈도(상대빈도)'
    p4.title.text = '여성 사용자 노출량 분포' + '(n=' + str(round(np.sum(selected_w_use_rate.sum(axis=1) > 0))) + ')'
    daylight_savings_start = bokeh.models.Span(location=start_date,
                                            dimension='height', line_color='red',
                                            line_dash='dashed', line_width=1)
    daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                    x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
    p4.renderers.extend([daylight_savings_start, daylight_savings_start_label])
    p4_table_data = [format(np.percentile(w_exposure[w_exposure > 0], 5),'.2E'), format(np.percentile(w_exposure[w_exposure > 0], 50),'.2E'),
                     format(np.percentile(w_exposure[w_exposure > 0], 75),'.2E'), format(np.percentile(w_exposure[w_exposure > 0], 90),'.2E'),
                     format(np.percentile(w_exposure[w_exposure > 0], 95),'.2E'), format(np.percentile(w_exposure[w_exposure > 0], 99),'.2E')]

    hist, edges = np.histogram(ax1, density=True, bins=50)



    p4.quad(top=hist,bottom=0, color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])

    p4.xaxis[0].formatter = FuncTickFormatter(code="""
      var str = tick.toString(); //get exponent
      var newStr = "";
      for (var i=0; i<10;i++)
      {
        var code = str.charCodeAt(i);
        switch(code) {
          case 45: // "-"
            newStr += "⁻";
            break;
          case 48: // "0"
            newStr +="⁰";
            break;
          case 49: // "1"
            newStr +="¹";
            break;
          case 50: // "2"
            newStr +="²";
            break;
          case 51: // "3"
            newStr +="³"
            break;
          case 52: // "4"
            newStr +="⁴"
            break;
          case 53: // "5"
            newStr +="⁵"
            break;                
          case 54: // "6"
            newStr +="⁶"
            break;
          case 55: // "7"
            newStr +="⁷"
            break;
          case 56: // "8"
            newStr +="⁸"
            break;
          case 57: // "9"
            newStr +="⁹"
            break;                         
        }
      }
      return 10+newStr;
    """)
    p4.axis.major_label_text_font = "sans"

    # 전체집단 노출 기여도 파이차트
    from math import pi
    from bokeh.palettes import RdYlBu
    name = []
    for i in range(len(selected_product_list)):
        for j in range(selected_alg_list.shape[0]):
            if selected_product_list[i] == selected_alg_list['code'].iloc[j]:
                name.append(selected_alg_list['category'].iloc[j] + selected_alg_list['use'].iloc[j]
                            + selected_alg_list['type_1'].iloc[j])

    ind_alg_sum = np.sum(alg_sum, axis=0)
    counts = pd.Series(list(ind_alg_sum), index=name)
    counts = counts.sort_values(ascending=False)

    counts = counts.head(10)
    data = pd.Series(counts).reset_index(name='value').rename(columns={'index': 'country'})

    if len(ind_alg_sum) == 1:
        data['color'] = ['#91bfdb']
    elif len(ind_alg_sum) == 2:
        data['color'] = ['#91bfdb', '#ffffbf']
    elif len(ind_alg_sum) < 10:
        data['color'] = RdYlBu[len(counts)]
    else:
        sum = counts.tail(len(counts) - 10).sum(axis=0)
        data.loc[10] = ['기타', sum]
        data['color'] = RdYlBu[len(counts) + 1]

    # data['angle'] = round((data['value'] / data['value'].sum() * 2 * pi),3)
    # data['value'] = round(data['value'],3)

    # percent = 100. * counts / counts.sum()

    # p5 = bokeh.plotting.figure(plot_width=720, plot_height=500, toolbar_location=None,
    #        tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    # from bokeh.transform import cumsum
    # p5.wedge(x=0, y=1, radius=0.4,
    #         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
    #         line_color="white", fill_color='color', legend_field='country', source=data)

    # p5.axis.axis_label = None
    # p5.axis.visible = False
    # p5.grid.grid_line_color = None

    #bokeh.plotting.show(p5)

    start_date = np.log10(np.percentile(inh_exposure [inh_exposure  > 0], 95))
    ax1 = np.log10(inh_exposure [inh_exposure  > 0])  # table
    ax1 = np.sort(ax1)
    p6 = bokeh.plotting.figure(sizing_mode="stretch_width", width=580, height=390)

    p6.xaxis.axis_label = '노출량(mg/m3)'
    p6.yaxis.axis_label = '빈도(상대빈도)'
    p6.title.text = '사용자 흡입 노출량 분포' + '(n=' + str(round(np.sum(np.array(selected_use_rate).sum(axis=1) > 0))) + ')'

    p6_table_data = [format(np.percentile(inh_exposure [inh_exposure  > 0], 5), '.2E'),
                     format(np.percentile(inh_exposure [inh_exposure  > 0], 50),'.2E'),
                     format(np.percentile(inh_exposure [inh_exposure  > 0], 75),'.2E'),
                     format(np.percentile(inh_exposure [inh_exposure  > 0], 90),'.2E'),
                     format(np.percentile(inh_exposure [inh_exposure  > 0], 95),'.2E'),
                     format(np.percentile(inh_exposure [inh_exposure  > 0], 99),'.2E')]

    hist, edges = np.histogram(ax1, density=True, bins=50)
    daylight_savings_start = bokeh.models.Span(location=start_date,
                                        dimension='height', line_color='red',
                                        line_dash='dashed', line_width=1)
    daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                    x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
    p6.renderers.extend([daylight_savings_start, daylight_savings_start_label])
    p6.quad(top=hist,bottom=0, color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])

    p6.xaxis[0].formatter = FuncTickFormatter(code="""
          var str = tick.toString(); //get exponent
          var newStr = "";
          for (var i=0; i<10;i++)
          {
            var code = str.charCodeAt(i);
            switch(code) {
              case 45: // "-"
                newStr += "⁻";
                break;
              case 48: // "0"
                newStr +="⁰";
                break;
              case 49: // "1"
                newStr +="¹";
                break;
              case 50: // "2"
                newStr +="²";
                break;
              case 51: // "3"
                newStr +="³"
                break;
              case 52: // "4"
                newStr +="⁴"
                break;
              case 53: // "5"
                newStr +="⁵"
                break;                
              case 54: // "6"
                newStr +="⁶"
                break;
              case 55: // "7"
                newStr +="⁷"
                break;
              case 56: // "8"
                newStr +="⁸"
                break;
              case 57: // "9"
                newStr +="⁹"
                break;                         
            }
          }
          return 10+newStr;
        """)
    p6.axis.major_label_text_font = "sans"

    inh_alg_sum = np.concatenate([m_inh_alg_sum, w_inh_alg_sum])
    ind_inh_alg_sum = np.sum(inh_alg_sum, axis=0)

    # 전체집단 흡입 노출 기여도 파이차트
    name = []
    for i in range(len(selected_product_list)):
        for j in range(selected_alg_list.shape[0]):
            if selected_product_list[i] == selected_alg_list['code'].iloc[j]:
                name.append(selected_alg_list['category'].iloc[j] + selected_alg_list['use'].iloc[j]
                            + selected_alg_list['type_1'].iloc[j])

    counts = pd.Series(list(ind_inh_alg_sum), index=name)
    counts = counts.sort_values(ascending=False)
    data2=pd.DataFrame(counts,columns=['value'])

    return alg_name, gen_alg1, gen_alg2, gen_alg3, gen_alg4, gen_alg5, gen_alg6,p2_table_data, p3_table_data,\
           p4_table_data, p6_table_data, p1, p2, p3, p4, data, p6, alg_sum, selected_product_list, p7, data2