from pathlib import Path
from tkinter.ttk import Style
import pandas as pd
import panel as pn
# import dask.dataframe as dd
import numpy as np
from io import BytesIO
import random
import plotly.express as px
import plotly.io as pio
import random
import plotly.graph_objects as go
import plotly.figure_factory as ff
import hvplot.pandas
import biocide
from scipy.integrate import odeint
from bokeh.models import FuncTickFormatter
import markdown
from math import pi
from bokeh.palettes import RdYlBu
import markdown
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import bokeh
from shapely.geometry import Polygon, LineString, Point
import os
import geopandas as gpd # GeoPandas(지오판다스)
# import rasterio
import shutil # shutil(shell utility, 쉘 유틸리티)
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import folium
import branca.colormap as cmp
import aqua
import rich
import time
from matplotlib.pyplot import legend
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
path = os.getcwd()
os.getcwd()

css = '''
.bk.panel-widget {
  border: None;
  font-size: 20px;
}

.button .bk-btn{
  font-size:20px;
  font-family: NanumBarunGothic;
}

.widget-button .bk-btn {
  font-size:20px;
  font-family: NanumBarunGothic;
}

.table .tabulator {
  font-size: 20px;
}

'''

pio.renderers.default='notebook'
pn.extension('tabulator','plotly','katex', 'mathjax',loading_spinner='dots', loading_color='#00aa41',sizing_mode='stretch_width',raw_css=[css] ,css_files=[pn.io.resources.CSS_URLS['font-awesome']])
pd.options.display.float_format = '{:.1E}'.format

list_1=pd.read_csv("604chemical_default.csv", thousands = ',')
cas_rn=list_1['CAS_Num']+" "+list_1['CHEM']
cas_rn_val=cas_rn.values
cas_rn_val=cas_rn_val.tolist()
cas_rn_val.insert(0,'')
optionss=cas_rn_val.copy()

select_cami=pn.widgets.Select(name="화학물질_리스트", options=optionss, value='', sizing_mode='fixed',width=500,css_classes=['panel-widget'])

chemi_input= pn.widgets.TextInput(name='화학물질 입력', sizing_mode='fixed',width=150,css_classes=['panel-widget'])
button3 = pn.widgets.Button(name='검색', button_type='primary',sizing_mode='fixed',width=150,css_classes=['button'])

@pn.depends(x=select_cami.param.value)
def selcet_input(x):
    if x=='':
        chemi_input.value=''
        select_input_widget=pn.Column(chemi_input,button3)
    else :
        x=str(x)
        x=x[:11]
        chemi_input.value=x
        select_input_widget=pn.Column(chemi_input,button3)
    return select_input_widget

options2=['inherently biodegradable','not biodegradable','readily biodegradable','readily biodegradable, failing 10-d window']
text_input = pn.widgets.TextInput(name='물질명', sizing_mode='fixed',width=550)
text_input2 = pn.widgets.TextInput(name='분자량 (g/mol)',sizing_mode='fixed',width=120)
text_input3 = pn.widgets.TextInput(name='녹는점(℃)', sizing_mode='fixed',width=120)
text_input4 = pn.widgets.TextInput(name='옥탄올-물 분배계수', sizing_mode='fixed',width=120)
text_input5 = pn.widgets.TextInput(name='증기압 (at 25℃) (Pa)', sizing_mode='fixed',width=120)
text_input6 = pn.widgets.TextInput(name='물용해도 (at 25℃) (㎎/L)', sizing_mode='fixed',width=120)
select_model=pn.widgets.Select(name='이분해성', options=options2, value='', sizing_mode='fixed')
text_input8 = pn.widgets.TextInput(name='Koc (L/㎏)', sizing_mode='fixed',width=120,margin=(0,0,20,10))
# text_input10 = pn.widgets.TextInput(name='PNEC', sizing_mode='fixed',width=120,margin=(0,0,20,10))
widget_box2=pn.Column(text_input, text_input2, text_input3, text_input4, text_input5,text_input6, select_model, text_input8,css_classes=['panel-widget'])

radio_group = pn.widgets.RadioBoxGroup(name='RadioBoxGroup', options=['수생태 환경노출량 산정', '인체노출량 산정'], inline=False,css_classes=['panel-widget'])
file_input = pn.widgets.FileInput(accept='.csv,.json',name='유통량 자료 업로드',sizing_mode='fixed')

radio_group3 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['수환경 예측환경농도 입력정보', '수생태 예측환경농도', '수환경 인체 간접노출평가 입력정보','수환경 인체 간접 노출량 산정결과'], sizing_mode='stretch_width', button_type='primary',margin=(0,0,50,0),css_classes=['widget-button'])

radio_group4 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group2', options=['입력정보확인', '누적노출분포', '제품별노출분포','제품별기여도'], button_type='success',margin=(0,0,50,0),css_classes=['widget-button'])

button = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed',css_classes=['button'])
button2 = pn.widgets.Button(name='Refresh', button_type='primary',sizing_mode='fixed',width=120,css_classes=['button'])


mark3=pn.pane.Markdown("### [제품 정보 입력]", sizing_mode='fixed', margin=(0, 0, 20, 0), style={'font-family': 'NanumBarunGothic','font-size':'25px'})
mark4=pn.pane.Markdown("### [물질량 정보 입력]", sizing_mode='fixed', margin=(0, 0, 20, 0), style={'font-family': 'NanumBarunGothic','font-size':'25px'})
mark5=pn.pane.Markdown("<br>")
file_download = pn.widgets.FileDownload(file='입력자료 템플릿.csv', filename='입력자료 템플릿.csv',sizing_mode='fixed',width=250,margin=(25,0,25,0))
# file_download2 = pn.widgets.FileDownload(file='사용량 자료 템플릿.csv', filename='사용량 자료 템플릿.csv',sizing_mode='fixed',width=250,margin=(25,0,25,0))
file_input = pn.widgets.FileInput(accept='.csv,.json',name='유통량 자료 업로드',sizing_mode='fixed')

value_option1='3.068'
value_option2='54.520'

radio_group2 = pn.widgets.RadioBoxGroup(name='RadioBoxGroup', options=['입력자료 갯수 5개 이하', '입력자료 업로드'], inline=False,css_classes=['panel-widget'])
options4=['','살균제류(소독제류)','구제제류']
select_option1=pn.widgets.Select(name='제품분류', options=options4, value='살균제류(소독제류)', sizing_mode='fixed')

options5=['','기피제','살균제', '살서제', '살조제', '살충제']
select_option2=pn.widgets.Select(name='제품유형', options=options5, value='살균제', sizing_mode='fixed')

options6=['','겔형,페이스트','고체형(타블릿,펠렛)','과립형','기타 - 거치형','기타 - 훈증형','분말형','액상','에어로솔','에어로솔폼','연무형','연소형','트리거','트리거폼','티슈,시트형']
select_option3=pn.widgets.Select(name='제품제형', options=options6, value='에어로솔', sizing_mode='fixed')

options7=['','실내','실외']
select_option4=pn.widgets.Select(name='사용장소', options=options7, value='실내', sizing_mode='fixed')

options8=['','분사','비분사']
select_option5=pn.widgets.Select(name='적용방식', options=options8, value='분사', sizing_mode='fixed')

options9=['','Leave-on','Rinse-off','Wipe-out']
select_option6=pn.widgets.Select(name='후처리방식', options=options9, value='Wipe-out', sizing_mode='fixed')

text_input11 = pn.widgets.TextInput(name='국내생산제품에 포함된 물질량 (톤/년)', value=value_option1,sizing_mode='fixed',width=120) ##
text_input12 = pn.widgets.TextInput(name='국내수입제품에 포함된 물질량 (톤/년)', value=value_option2,sizing_mode='fixed',width=120) ##
#######################################################################################################################################################################

select_option7=pn.widgets.Select(name='제품분류', options=options4, value='살균제류(소독제류)', sizing_mode='fixed')

select_option8=pn.widgets.Select(name='제품유형', options=options5, value='살균제', sizing_mode='fixed')

options10=['','겔형,페이스트','고체형(타블릿,펠렛)','과립형','기타 - 거치형','기타 - 훈증형','분말형','액상','에어로솔','에어로솔폼','연무형','연소형','트리거','트리거폼','티슈,시트형']
select_option9=pn.widgets.Select(name='제품제형', options=options10, value='액상', sizing_mode='fixed')

select_option10=pn.widgets.Select(name='사용장소', options=options7, value='실외', sizing_mode='fixed')

select_option11=pn.widgets.Select(name='적용방식', options=options8, value='비분사', sizing_mode='fixed')

select_option12=pn.widgets.Select(name='후처리방식', options=options9, value='Leave-on', sizing_mode='fixed')

text_input13 = pn.widgets.TextInput(name='국내생산제품에 포함된 물질량 (톤/년)', value=value_option1,sizing_mode='fixed',width=120) ##
text_input14 = pn.widgets.TextInput(name='국내수입제품에 포함된 물질량 (톤/년)', value=value_option2,sizing_mode='fixed',width=120) ##

#######################################################################################################################################################################

select_option13=pn.widgets.Select(name='제품분류', options=options4, value='살균제류(소독제류)', sizing_mode='fixed')

select_option14=pn.widgets.Select(name='제품유형', options=options5, value='살균제', sizing_mode='fixed')

options11=['','겔형,페이스트','고체형(타블릿,펠렛)','과립형','기타 - 거치형','기타 - 훈증형','분말형','액상','에어로솔','에어로솔폼','연무형','연소형','트리거','트리거폼','티슈,시트형']
select_option15=pn.widgets.Select(name='제품제형', options=options11, value='에어로솔', sizing_mode='fixed')


select_option16=pn.widgets.Select(name='사용장소', options=options7, value='실내', sizing_mode='fixed')

select_option17=pn.widgets.Select(name='적용방식', options=options8, value='분사', sizing_mode='fixed')

select_option18=pn.widgets.Select(name='후처리방식', options=options9, value='Wipe-out', sizing_mode='fixed')

text_input15 = pn.widgets.TextInput(name='국내생산제품에 포함된 물질량 (톤/년)', value=value_option1,sizing_mode='fixed',width=120) ##
text_input16 = pn.widgets.TextInput(name='국내수입제품에 포함된 물질량 (톤/년)', value=value_option2,sizing_mode='fixed',width=120) ##

#######################################################################################################################################################################

select_option19=pn.widgets.Select(name='제품분류', options=options4, value='살균제류(소독제류)', sizing_mode='fixed')

select_option20=pn.widgets.Select(name='제품유형', options=options5, value='살균제', sizing_mode='fixed')

options12=['','겔형,페이스트','고체형(타블릿,펠렛)','과립형','기타 - 거치형','기타 - 훈증형','분말형','액상','에어로솔','에어로솔폼','연무형','연소형','트리거','트리거폼','티슈,시트형']
select_option21=pn.widgets.Select(name='제품제형', options=options12, value='에어로솔', sizing_mode='fixed')

select_option22=pn.widgets.Select(name='사용장소', options=options7, value='실내', sizing_mode='fixed')

select_option23=pn.widgets.Select(name='적용방식', options=options8, value='분사', sizing_mode='fixed')

select_option24=pn.widgets.Select(name='후처리방식', options=options9, value='Wipe-out', sizing_mode='fixed')

text_input17 = pn.widgets.TextInput(name='국내생산제품에 포함된 물질량 (톤/년)', value=value_option1,sizing_mode='fixed',width=120) ##
text_input18 = pn.widgets.TextInput(name='국내수입제품에 포함된 물질량 (톤/년)', value=value_option2,sizing_mode='fixed',width=120) ##

#######################################################################################################################################################################

select_option25=pn.widgets.Select(name='제품분류', options=options4, value='살균제류(소독제류)', sizing_mode='fixed')

select_option26=pn.widgets.Select(name='제품유형', options=options5, value='살균제', sizing_mode='fixed')

options13=['','겔형,페이스트','고체형(타블릿,펠렛)','과립형','기타 - 거치형','기타 - 훈증형','분말형','액상','에어로솔','에어로솔폼','연무형','연소형','트리거','트리거폼','티슈,시트형']
select_option27=pn.widgets.Select(name='제품제형', options=options13, value='에어로솔', sizing_mode='fixed')

select_option28=pn.widgets.Select(name='사용장소', options=options7, value='실내', sizing_mode='fixed')

select_option29=pn.widgets.Select(name='적용방식', options=options8, value='분사', sizing_mode='fixed')

select_option30=pn.widgets.Select(name='후처리방식', options=options9, value='Wipe-out', sizing_mode='fixed')

text_input19 = pn.widgets.TextInput(name='국내생산제품에 포함된 물질량 (톤/년)', value=value_option1,sizing_mode='fixed',width=120) ##
text_input20 = pn.widgets.TextInput(name='국내수입제품에 포함된 물질량 (톤/년)', value=value_option2,sizing_mode='fixed',width=120) ##
#######################################################################################################################################################################
text_input21 = pn.widgets.TextInput(name='입력자료 갯수 입력', value='',sizing_mode='fixed',width=120) ##
button4=pn.widgets.Button(name='입력', button_type='primary',sizing_mode='fixed',width=120,css_classes=['button'])
input_self=pn.Column(text_input21,button4)
@pn.depends(button4.param.clicks)
def input_kind(_):
    if text_input21.value =='':
        tabs=pn.Column()
    else:
        ######
        if chemi_input.value =='007173-51-5':
            text_input11.value='3.068'
            text_input13.value='3.068'
            text_input15.value='3.068'
            text_input17.value='3.068'
            text_input19.value='3.068'
            text_input12.value='54.520'
            text_input14.value='54.520'
            text_input16.value='54.520'
            text_input18.value='54.520'
            text_input20.value='54.520'
            if text_input21.value =='1':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='3':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                dynamic=True
                            )
            elif text_input21.value =='4':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)),
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                dynamic=True
                            )
            elif text_input21.value =='5':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                ('다섯번째_제품',pn.Column(mark3,select_option25,select_option26,select_option27,select_option28,select_option29,select_option30,mark5,mark4,text_input19,text_input20)), 
                                dynamic=True
                            )
        elif chemi_input.value =='001222-05-5':
            text_input11.value='0'
            text_input13.value='0'
            text_input15.value='0'
            text_input17.value='0'
            text_input19.value='0'
            text_input12.value='215.753'
            text_input14.value='215.753'
            text_input16.value='215.753'
            text_input18.value='215.753'
            text_input20.value='215.753'
            if text_input21.value =='1':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='3':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                dynamic=True
                            )
            elif text_input21.value =='4':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)),
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                dynamic=True
                            )
            elif text_input21.value =='5':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                ('다섯번째_제품',pn.Column(mark3,select_option25,select_option26,select_option27,select_option28,select_option29,select_option30,mark5,mark4,text_input19,text_input20)), 
                                dynamic=True
                            )

        elif chemi_input.value =='002634-33-5':
            text_input11.value='139.683'
            text_input13.value='139.683'
            text_input15.value='139.683'
            text_input17.value='139.683'
            text_input19.value='139.683'
            text_input12.value='24.918'
            text_input14.value='24.918'
            text_input16.value='24.918'
            text_input18.value='24.918'
            text_input20.value='24.918'
            if text_input21.value =='1':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='3':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                dynamic=True
                            )
            elif text_input21.value =='4':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)),
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                dynamic=True
                            )
            elif text_input21.value =='5':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                ('다섯번째_제품',pn.Column(mark3,select_option25,select_option26,select_option27,select_option28,select_option29,select_option30,mark5,mark4,text_input19,text_input20)), 
                                dynamic=True
                            )        

        else: 
        ######
            if text_input21.value =='1':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='2':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                dynamic=True
                            )
            elif text_input21.value =='3':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                dynamic=True
                            )
            elif text_input21.value =='4':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)),
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                dynamic=True
                            )
            elif text_input21.value =='5':
                tabs=pn.Tabs(
                                ('첫번째_제품',pn.Column(mark3,select_option1,select_option2,select_option3,select_option4,select_option5,select_option6,mark5,mark4,text_input11,text_input12)),
                                ('두번째_제품',pn.Column(mark3,select_option7,select_option8,select_option9,select_option10,select_option11,select_option12,mark5,mark4,text_input13,text_input14)),
                                ('세번째_제품',pn.Column(mark3,select_option13,select_option14,select_option15,select_option16,select_option17,select_option18,mark5,mark4,text_input15,text_input16)), 
                                ('네번째_제품',pn.Column(mark3,select_option19,select_option20,select_option21,select_option22,select_option23,select_option24,mark5,mark4,text_input17,text_input18)), 
                                ('다섯번째_제품',pn.Column(mark3,select_option25,select_option26,select_option27,select_option28,select_option29,select_option30,mark5,mark4,text_input19,text_input20)), 
                                dynamic=True
                            )
    return tabs
@pn.depends(x=radio_group2.param.value)
def radio_option2(x):
    if x == '입력자료 갯수 5개 이하':
        widget_box3=pn.Column(input_self,mark5,input_kind,css_classes=['panel-widget'])
    elif x =='입력자료 업로드':
        widget_box3=pn.Column(file_download,file_input,css_classes=['panel-widget'])
    return widget_box3
@pn.depends(x=radio_group.param.value)
def radio_option(x):
    marks=pn.pane.Markdown("## <br> ■  입력 자료 갯수 선택 <br>",style={'font-family': 'NanumBarunGothic','font-size':'20px'})
    if x == '수생태 환경노출량 산정':
        widget_box3=pn.Column(marks,radio_group2,mark5,radio_option2,mark5)
    elif x =='인체노출량 산정':
        widget_box3=pn.Column()
    return widget_box3
mark=pn.pane.Markdown('<br>')
mark2=pn.pane.Markdown("#### 본 프로그램에서 사용가능한 CAS넘버는 아래의 리스트에서 확인 가능합니다.", style={'font-family': 'NanumBarunGothic','font-size':'25px'})

widget_box=pn.Column(pn.pane.Markdown('<br>'),pn.pane.Markdown('## ■ 물성정보확인 (물성정보 수정시, 직접입력)', style={'font-family': 'NanumBarunGothic','font-size':'20px'}),widget_box2,pn.pane.Markdown('## ■ 생태 및 인체 노출량 산정방식 선택', style={'font-family': 'NanumBarunGothic','font-size':'20px'}),radio_group,radio_option,button)

@pn.depends(button3.param.clicks)
def search_chemi(_):
    if chemi_input.value =='':
        side=pn.Column()
    else:
        @pn.depends(x=chemi_input.param.value)
        def widget_value(x):
            chemi_df=list_1.copy()
            if x == '':
                options=['','','','','','','']
            else:
                if (chemi_df['CAS_Num']==x).any():
                    values=np.array(chemi_df[chemi_df['CAS_Num'] == x]).flatten()
                    options=[str(values[0]),str(round(values[2],2)),str(round(values[3],2)),str(format(values[4],'.2E')),str(format(values[5],'.2E')),str(round(values[6],2)),str(format(values[7],'.2E')),str(values[8])]
            return options
        @pn.depends(x=chemi_input.param.value)
        def side_area(x):
            if x == '':
                side=pn.Column()
            else:
                options=widget_value(x)
                text_input.value=options[0]
                text_input2.value=options[1]
                text_input3.value=options[2]
                text_input4.value=options[3]
                text_input5.value=options[4]
                text_input6.value=options[5]
                select_model.value=options[7]
                text_input8.value=options[6]
                side=pn.Row(widget_box,button2)
            return side
        x=chemi_input.value
        side=side_area(x)
    return side
def refresh(event):
    text_input.value=''
    text_input2.value=''
    text_input3.value=''
    text_input4.value='' 
    text_input5.value='' 
    text_input6.value='' 
    select_model.value=''    
    text_input8.value=''  
button2.on_click(refresh)

mark7=pn.pane.Markdown("#### - 생산제품중 물질량 : 연간 국내에서 생산된 제품에 포함된 물질의 총량 <br>  - 수입제품중 물질량 : 국외에서 생산되어 연간 국내로 수입된 제품에 포함된 물질의 총량 <br>   - 국내유통 물질량 : 국내생산제품에 포함된 총 물질량과 국내수입제품에 포함된 총 물질량의 합 (제품 사용단계에서 모두 환경으로 배출된다고 가정) ", style={'font-family': 'NanumBarunGothic','font-size':'20px'})

mark8=pn.pane.Markdown("#### ■ 물질 제조단계 및 제품 생산단계 환경배출량 <br>   - 물질의 제조단계 및 제품의 생산단계에서 배출되는 물질의 양 (제조 및 생산 공정별 배출계수 반영) <br>  -  공정에 대한 정보가 없는 경우 보수적으로 접근 : <br> 물질 제조단계에서 배출계수 ERC1 (대기 5%, 수계 6%, 토양 0.01%), 제품 생산단계에서 배출계수 ERC2 (2.5%, 2%, 0.01%) <br><br> ■ 제품 사용단계 환경배출량 <br>  -  제품에 포함된 물질량에 제품별 배출계수를 곱해 제품 사용시 환경으로 배출되는 물질량을 매체별로 계산하고, 전체 제품군에 대해 매체별 배출량을 합산한 값 <br><br> ■ 1인 1일 평균 물질사용량 : 연간 국내에서 유통되는 제품에 포함된 총 물질량을 국내 전체 인구수와 연간 사용일수(365일)로 나눈 값", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
@pn.depends(x=radio_group2.param.value)
def make_df_2(x):
    chemi_df=pd.read_csv('05_DB_EF_Table.csv')
    if x == '입력자료 갯수 5개 이하':
        if text_input21.value =='1':
            if (chemi_df['P_Category_01']==select_option1.value).any() & (chemi_df['P_Category_02']==select_option2.value).any() & (chemi_df['P_Type']==select_option3.value).any() & (chemi_df['U_Space']==select_option4.value).any() & (chemi_df['U_Type']==select_option5.value).any() & (chemi_df['U_After']==select_option6.value).any() :
                values=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option1.value) 
                & (chemi_df['P_Category_02']==select_option2.value) 
                & (chemi_df['P_Type']==select_option3.value) 
                & (chemi_df['U_Space']==select_option4.value) 
                & (chemi_df['U_Type']==select_option5.value) 
                & (chemi_df['U_After']==select_option6.value) ])).flatten()
            df2=pd.DataFrame({'CAS_RN':[chemi_input.value],
                                'P_Category_01':[select_option1.value],
                                'P_Category_02':[select_option2.value],
                                'P_Type':[select_option3.value],
                                'U_Space':[select_option4.value],
                                'U_Type':[select_option5.value],
                                'U_After':[select_option6.value],
                                'Chem_Amount_Domestic':[float(text_input11.value)],
                                'Chem_Amount_Import':[float(text_input12.value)],
                                'Chem_Amount_Total':[float(text_input11.value)+float(text_input12.value)],
                                'EF_P_Air':[float(values[3])],
                                'EF_P_Water':[float(values[4])],
                                'EF_P_Soil':[float(values[5])],
                                'EF_U_Air':[float(values[9])],
                                'EF_U_Water':[float(values[10])],
                                'EF_U_Soil':[float(values[11])]})  

        elif text_input21.value =='2':
            if (chemi_df['P_Category_01']==select_option1.value).any() & (chemi_df['P_Category_02']==select_option2.value).any() & (chemi_df['P_Type']==select_option3.value).any() & (chemi_df['U_Space']==select_option4.value).any() & (chemi_df['U_Type']==select_option5.value).any() & (chemi_df['U_After']==select_option6.value).any() :
                values=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option1.value) 
                & (chemi_df['P_Category_02']==select_option2.value) 
                & (chemi_df['P_Type']==select_option3.value) 
                & (chemi_df['U_Space']==select_option4.value) 
                & (chemi_df['U_Type']==select_option5.value) 
                & (chemi_df['U_After']==select_option6.value) ])).flatten()
            if (chemi_df['P_Category_01']==select_option7.value).any() & (chemi_df['P_Category_02']==select_option8.value).any() & (chemi_df['P_Type']==select_option9.value).any() & (chemi_df['U_Space']==select_option10.value).any() & (chemi_df['U_Type']==select_option11.value).any() & (chemi_df['U_After']==select_option12.value).any() :
                values2=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option7.value) 
                & (chemi_df['P_Category_02']==select_option8.value) 
                & (chemi_df['P_Type']==select_option9.value) 
                & (chemi_df['U_Space']==select_option10.value) 
                & (chemi_df['U_Type']==select_option11.value) 
                & (chemi_df['U_After']==select_option12.value)])).flatten()
            df2=pd.DataFrame({'CAS_RN':[chemi_input.value,chemi_input.value],
                                'P_Category_01':[select_option1.value,select_option7.value],
                                'P_Category_02':[select_option2.value,select_option8.value],
                                'P_Type':[select_option3.value,select_option9.value],
                                'U_Space':[select_option4.value,select_option10.value],
                                'U_Type':[select_option5.value,select_option11.value],
                                'U_After':[select_option6.value,select_option12.value],
                                'Chem_Amount_Domestic':[float(text_input11.value),float(text_input13.value)],
                                'Chem_Amount_Import':[float(text_input12.value),float(text_input14.value)],
                                'Chem_Amount_Total':[float(text_input11.value)+float(text_input12.value),float(text_input13.value)+float(text_input14.value)],
                                'EF_P_Air':[float(values[3]),float(values2[3])],
                                'EF_P_Water':[float(values[4]),float(values2[4])],
                                'EF_P_Soil':[float(values[5]),float(values2[5])],
                                'EF_U_Air':[float(values[9]),float(values2[9])],
                                'EF_U_Water':[float(values[10]),float(values2[10])],
                                'EF_U_Soil':[float(values[11]),float(values2[11])]})     

        elif text_input21.value =='3':
            if (chemi_df['P_Category_01']==select_option1.value).any() & (chemi_df['P_Category_02']==select_option2.value).any() & (chemi_df['P_Type']==select_option3.value).any() & (chemi_df['U_Space']==select_option4.value).any() & (chemi_df['U_Type']==select_option5.value).any() & (chemi_df['U_After']==select_option6.value).any() :
                values=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option1.value) 
                & (chemi_df['P_Category_02']==select_option2.value) 
                & (chemi_df['P_Type']==select_option3.value) 
                & (chemi_df['U_Space']==select_option4.value) 
                & (chemi_df['U_Type']==select_option5.value) 
                & (chemi_df['U_After']==select_option6.value) ])).flatten()
            if (chemi_df['P_Category_01']==select_option7.value).any() & (chemi_df['P_Category_02']==select_option8.value).any() & (chemi_df['P_Type']==select_option9.value).any() & (chemi_df['U_Space']==select_option10.value).any() & (chemi_df['U_Type']==select_option11.value).any() & (chemi_df['U_After']==select_option12.value).any() :
                values2=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option7.value) 
                & (chemi_df['P_Category_02']==select_option8.value) 
                & (chemi_df['P_Type']==select_option9.value) 
                & (chemi_df['U_Space']==select_option10.value) 
                & (chemi_df['U_Type']==select_option11.value) 
                & (chemi_df['U_After']==select_option12.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option13.value).any() & (chemi_df['P_Category_02']==select_option14.value).any() & (chemi_df['P_Type']==select_option15.value).any() & (chemi_df['U_Space']==select_option16.value).any() & (chemi_df['U_Type']==select_option17.value).any() & (chemi_df['U_After']==select_option18.value).any() :
                values3=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option13.value) 
                & (chemi_df['P_Category_02']==select_option14.value) 
                & (chemi_df['P_Type']==select_option15.value) 
                & (chemi_df['U_Space']==select_option16.value) 
                & (chemi_df['U_Type']==select_option17.value) 
                & (chemi_df['U_After']==select_option18.value)])).flatten()
            df2=pd.DataFrame({'CAS_RN':[chemi_input.value,chemi_input.value,chemi_input.value],
                                'P_Category_01':[select_option1.value,select_option7.value,select_option13.value],
                                'P_Category_02':[select_option2.value,select_option8.value,select_option14.value],
                                'P_Type':[select_option3.value,select_option9.value,select_option15.value],
                                'U_Space':[select_option4.value,select_option10.value,select_option16.value],
                                'U_Type':[select_option5.value,select_option11.value,select_option17.value],
                                'U_After':[select_option6.value,select_option12.value,select_option18.value],
                                'Chem_Amount_Domestic':[float(text_input11.value),float(text_input13.value),float(text_input15.value)],
                                'Chem_Amount_Import':[float(text_input12.value),float(text_input14.value),float(text_input16.value)],
                                'Chem_Amount_Total':[float(text_input11.value)+float(text_input12.value),float(text_input13.value)+float(text_input14.value),float(text_input15.value)+float(text_input16.value)],
                                'EF_P_Air':[float(values[3]),float(values2[3]),float(values3[3])],
                                'EF_P_Water':[float(values[4]),float(values2[4]),float(values3[4])],
                                'EF_P_Soil':[float(values[5]),float(values2[5]),float(values3[5])],
                                'EF_U_Air':[float(values[9]),float(values2[9]),float(values3[9])],
                                'EF_U_Water':[float(values[10]),float(values2[10]),float(values3[10])],
                                'EF_U_Soil':[float(values[11]),float(values2[11]),float(values3[11])]})   
        elif text_input21.value =='4':
            if (chemi_df['P_Category_01']==select_option1.value).any() & (chemi_df['P_Category_02']==select_option2.value).any() & (chemi_df['P_Type']==select_option3.value).any() & (chemi_df['U_Space']==select_option4.value).any() & (chemi_df['U_Type']==select_option5.value).any() & (chemi_df['U_After']==select_option6.value).any() :
                values=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option1.value) 
                & (chemi_df['P_Category_02']==select_option2.value) 
                & (chemi_df['P_Type']==select_option3.value) 
                & (chemi_df['U_Space']==select_option4.value) 
                & (chemi_df['U_Type']==select_option5.value) 
                & (chemi_df['U_After']==select_option6.value) ])).flatten()
            if (chemi_df['P_Category_01']==select_option7.value).any() & (chemi_df['P_Category_02']==select_option8.value).any() & (chemi_df['P_Type']==select_option9.value).any() & (chemi_df['U_Space']==select_option10.value).any() & (chemi_df['U_Type']==select_option11.value).any() & (chemi_df['U_After']==select_option12.value).any() :
                values2=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option7.value) 
                & (chemi_df['P_Category_02']==select_option8.value) 
                & (chemi_df['P_Type']==select_option9.value) 
                & (chemi_df['U_Space']==select_option10.value) 
                & (chemi_df['U_Type']==select_option11.value) 
                & (chemi_df['U_After']==select_option12.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option13.value).any() & (chemi_df['P_Category_02']==select_option14.value).any() & (chemi_df['P_Type']==select_option15.value).any() & (chemi_df['U_Space']==select_option16.value).any() & (chemi_df['U_Type']==select_option17.value).any() & (chemi_df['U_After']==select_option18.value).any() :
                values3=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option13.value) 
                & (chemi_df['P_Category_02']==select_option14.value) 
                & (chemi_df['P_Type']==select_option15.value) 
                & (chemi_df['U_Space']==select_option16.value) 
                & (chemi_df['U_Type']==select_option17.value) 
                & (chemi_df['U_After']==select_option18.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option19.value).any() & (chemi_df['P_Category_02']==select_option20.value).any() & (chemi_df['P_Type']==select_option21.value).any() & (chemi_df['U_Space']==select_option22.value).any() & (chemi_df['U_Type']==select_option23.value).any() & (chemi_df['U_After']==select_option24.value).any() :
                values4=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option19.value) 
                & (chemi_df['P_Category_02']==select_option20.value) 
                & (chemi_df['P_Type']==select_option21.value) 
                & (chemi_df['U_Space']==select_option22.value) 
                & (chemi_df['U_Type']==select_option23.value) 
                & (chemi_df['U_After']==select_option24.value)])).flatten()
            df2=pd.DataFrame({'CAS_RN':[chemi_input.value,chemi_input.value,chemi_input.value,chemi_input.value],
                                'P_Category_01':[select_option1.value,select_option7.value,select_option13.value,select_option19.value],
                                'P_Category_02':[select_option2.value,select_option8.value,select_option14.value,select_option20.value],
                                'P_Type':[select_option3.value,select_option9.value,select_option15.value,select_option21.value],
                                'U_Space':[select_option4.value,select_option10.value,select_option16.value,select_option22.value],
                                'U_Type':[select_option5.value,select_option11.value,select_option17.value,select_option23.value],
                                'U_After':[select_option6.value,select_option12.value,select_option18.value,select_option24.value],
                                'Chem_Amount_Domestic':[float(text_input11.value),float(text_input13.value),float(text_input15.value),float(text_input17.value)],
                                'Chem_Amount_Import':[float(text_input12.value),float(text_input14.value),float(text_input16.value),float(text_input18.value)],
                                'Chem_Amount_Total':[float(text_input11.value)+float(text_input12.value),float(text_input13.value)+float(text_input14.value),float(text_input15.value)+float(text_input16.value),float(text_input17.value)+float(text_input18.value)],
                                'EF_P_Air':[float(values[3]),float(values2[3]),float(values3[3]),float(values4[3])],
                                'EF_P_Water':[float(values[4]),float(values2[4]),float(values3[4]),float(values4[4])],
                                'EF_P_Soil':[float(values[5]),float(values2[5]),float(values3[5]),float(values4[5])],
                                'EF_U_Air':[float(values[9]),float(values2[9]),float(values3[9]),float(values4[9])],
                                'EF_U_Water':[float(values[10]),float(values2[10]),float(values3[10]),float(values4[10])],
                                'EF_U_Soil':[float(values[11]),float(values2[11]),float(values3[11]),float(values4[11])]})   
        elif text_input21.value =='5':
            if (chemi_df['P_Category_01']==select_option1.value).any() & (chemi_df['P_Category_02']==select_option2.value).any() & (chemi_df['P_Type']==select_option3.value).any() & (chemi_df['U_Space']==select_option4.value).any() & (chemi_df['U_Type']==select_option5.value).any() & (chemi_df['U_After']==select_option6.value).any() :
                values=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option1.value) 
                & (chemi_df['P_Category_02']==select_option2.value) 
                & (chemi_df['P_Type']==select_option3.value) 
                & (chemi_df['U_Space']==select_option4.value) 
                & (chemi_df['U_Type']==select_option5.value) 
                & (chemi_df['U_After']==select_option6.value) ])).flatten()
            if (chemi_df['P_Category_01']==select_option7.value).any() & (chemi_df['P_Category_02']==select_option8.value).any() & (chemi_df['P_Type']==select_option9.value).any() & (chemi_df['U_Space']==select_option10.value).any() & (chemi_df['U_Type']==select_option11.value).any() & (chemi_df['U_After']==select_option12.value).any() :
                values2=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option7.value) 
                & (chemi_df['P_Category_02']==select_option8.value) 
                & (chemi_df['P_Type']==select_option9.value) 
                & (chemi_df['U_Space']==select_option10.value) 
                & (chemi_df['U_Type']==select_option11.value) 
                & (chemi_df['U_After']==select_option12.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option13.value).any() & (chemi_df['P_Category_02']==select_option14.value).any() & (chemi_df['P_Type']==select_option15.value).any() & (chemi_df['U_Space']==select_option16.value).any() & (chemi_df['U_Type']==select_option17.value).any() & (chemi_df['U_After']==select_option18.value).any() :
                values3=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option13.value) 
                & (chemi_df['P_Category_02']==select_option14.value) 
                & (chemi_df['P_Type']==select_option15.value) 
                & (chemi_df['U_Space']==select_option16.value) 
                & (chemi_df['U_Type']==select_option17.value) 
                & (chemi_df['U_After']==select_option18.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option19.value).any() & (chemi_df['P_Category_02']==select_option20.value).any() & (chemi_df['P_Type']==select_option21.value).any() & (chemi_df['U_Space']==select_option22.value).any() & (chemi_df['U_Type']==select_option23.value).any() & (chemi_df['U_After']==select_option24.value).any() :
                values4=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option19.value) 
                & (chemi_df['P_Category_02']==select_option20.value) 
                & (chemi_df['P_Type']==select_option21.value) 
                & (chemi_df['U_Space']==select_option22.value) 
                & (chemi_df['U_Type']==select_option23.value) 
                & (chemi_df['U_After']==select_option24.value)])).flatten()
            if (chemi_df['P_Category_01']==select_option25.value).any() & (chemi_df['P_Category_02']==select_option26.value).any() & (chemi_df['P_Type']==select_option27.value).any() & (chemi_df['U_Space']==select_option28.value).any() & (chemi_df['U_Type']==select_option29.value).any() & (chemi_df['U_After']==select_option30.value).any() :
                values5=np.array((chemi_df.loc[(chemi_df['P_Category_01']==select_option25.value) 
                & (chemi_df['P_Category_02']==select_option26.value) 
                & (chemi_df['P_Type']==select_option27.value) 
                & (chemi_df['U_Space']==select_option28.value) 
                & (chemi_df['U_Type']==select_option29.value) 
                & (chemi_df['U_After']==select_option30.value)])).flatten()
            df2=pd.DataFrame({'CAS_RN':[chemi_input.value,chemi_input.value,chemi_input.value,chemi_input.value,chemi_input.value],
                                'P_Category_01':[select_option1.value,select_option7.value,select_option13.value,select_option19.value,select_option25.value],
                                'P_Category_02':[select_option2.value,select_option8.value,select_option14.value,select_option20.value,select_option26.value],
                                'P_Type':[select_option3.value,select_option9.value,select_option15.value,select_option21.value,select_option27.value],
                                'U_Space':[select_option4.value,select_option10.value,select_option16.value,select_option22.value,select_option28.value],
                                'U_Type':[select_option5.value,select_option11.value,select_option17.value,select_option23.value,select_option29.value],
                                'U_After':[select_option6.value,select_option12.value,select_option18.value,select_option24.value,select_option30.value],
                                'Chem_Amount_Domestic':[float(text_input11.value),float(text_input13.value),float(text_input15.value),float(text_input17.value),float(text_input19.value)],
                                'Chem_Amount_Import':[float(text_input12.value),float(text_input14.value),float(text_input16.value),float(text_input18.value),float(text_input20.value)],
                                'Chem_Amount_Total':[float(text_input11.value)+float(text_input12.value),float(text_input13.value)+float(text_input14.value),float(text_input15.value)+float(text_input16.value),float(text_input17.value)+float(text_input18.value),float(text_input19.value)+float(text_input20.value)],
                                'EF_P_Air':[float(values[3]),float(values2[3]),float(values3[3]),float(values4[3]),float(values5[3])],
                                'EF_P_Water':[float(values[4]),float(values2[4]),float(values3[4]),float(values4[4]),float(values5[4])],
                                'EF_P_Soil':[float(values[5]),float(values2[5]),float(values3[5]),float(values4[5]),float(values5[5])],
                                'EF_U_Air':[float(values[9]),float(values2[9]),float(values3[9]),float(values4[9]),float(values5[9])],
                                'EF_U_Water':[float(values[10]),float(values2[10]),float(values3[10]),float(values4[10]),float(values5[10])],
                                'EF_U_Soil':[float(values[11]),float(values2[11]),float(values3[11]),float(values4[11]),float(values5[11])]})   
    elif x =='입력자료 업로드':
        stock_file = BytesIO()
        stock_file.write(file_input.value)
        stock_file.seek(0)  
        df2 = pd.read_csv(stock_file ,encoding='cp949')
    return df2


@pn.depends(button.param.clicks)
def calculate_A_batch(_):
    if chemi_input.value =='':
        tabs=pn.Column(pn.pane.JPG('수생태 노출량 평가 모델 첫 화면_new.jpg',height=460,width=760,margin=(0,0,50,0)))
        tabs.background_color="#ffffff"
    else:
        if radio_group.value=='수생태 환경노출량 산정':
            """
            Created on Mon Jun 27 15:37:19 2022

            @author: gwyoo & dykwak
            """

            # 물질선택, 제품 중 농도 입력
            chemical = chemi_input.value #'007173-51-5'
            mark2=pn.pane.Markdown("---")
            asq=radio_group2.value

            df = make_df_2(asq)
            df = df.set_index('CAS_RN')

            df2=df.copy()
            df2.columns=['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식','생산제품중_물질량 (톤/년)','수입제품중_물질량 (톤/년)','총 국내유통_ 물질량 (톤/년)','배출계수_대기','배출계수_수계','배출계수_토양','배출계수_대기_사용','배출계수_수계','배출계수_토양']
            df2=df2[['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식','생산제품중_물질량 (톤/년)','수입제품중_물질량 (톤/년)','총 국내유통_ 물질량 (톤/년)']]

            df3=df.copy()
            df3.columns=['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식','생산제품중_물질량 (톤/년)','수입제품중_물질량 (톤/년)','총 국내유통_ 물질량 (톤/년)','대기','수계','토양','대기_사용','수계_사용','토양_사용']
            df3_t=df3[['대기','수계','토양','대기_사용','수계_사용','토양_사용']]
            df3_t.columns=['대기','수계','토양','대기_사용','수계_사용','토양_사용']


            flow_1=pn.Column(pn.pane.Markdown("## Ⅰ-1. 하수종말처리장 방류지점의 예측환경농도(PEC<sub>local_water</sub>) 산정방법", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.pane.JPG('1page_new.jpg',height=470,width=600,margin=(0,0,50,0)))

            #STP 제거율 계산
            data = pd.read_csv("STP_CAL.csv")

            if select_model.value=='readily biodegradable':
                kdeg_t=1
            elif select_model.value=='inherently biodegradable':
                kdeg_t=2
            elif select_model.value=='not biodegradable':
                kdeg_t=3
            elif select_model.value=='readily biodegradable, failing 10-d window':
                kdeg_t=4

            Kdeg_int = int(kdeg_t)
            Kow_int = float(text_input4.value)

            if math.log10(Kow_int) < 0 :
                logKow = 0
            elif math.log10(Kow_int) > 6 :
                logKow = 6
            else:
                logKow = int(math.log10(Kow_int))

            vaporpressure = float(text_input5.value)
            molcularweight = float(text_input2.value)
            watersolubility = float(text_input6.value)

            Henry = vaporpressure / (molcularweight*watersolubility)

            if math.log10(Henry) < -4 :
                logH = -4
            elif math.log10(Henry) > 5 :
                logH = 5
            else:
                logH = int(math.log10(Henry))

            result = data[(data.iloc[:,0] == Kdeg_int+1) & (data.iloc[:, 1] == logH) & (data.iloc[:, 2] == logKow)] 

            out_STP_pecent = result.iloc[:,4].values[0]
            STP_removal_pecent = 100 - out_STP_pecent

            df_data=df2.loc[[chemical]]
            df_data_2=df_data.copy()
            df_data_2=df_data_2.reset_index(drop=True)
            df_data_2=df_data_2.set_index('분류')
            tabulator_editors = {
                '살생물제품유형': None,
                '제형': None,
                '사용장소': None,
                '적용방식': None,
                '후처리방식': None,
                '생산제품중_물질량 (톤/년)':None,
                '수입제품중_물질량 (톤/년)':None,
                '총 국내유통_ 물질량 (톤/년)':None,
            }
            table_a=pn.Column(pn.pane.Markdown("## Ⅰ-2. 물질함유 제품정보 ("+str(chemical)+")",style={'font-family': 'NanumBarunGothic','font-size':'20px'}),mark7,pn.widgets.Tabulator(df_data_2,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))

            df3_t=df3_t.loc[[chemical]]
            df3_s=df_data.copy()
            df3_s=df3_s[['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식']]
            df3_s_t=pd.concat([df3_s,df3_t],axis=1)
            df3_s_t=df3_s_t.reset_index(drop=True)
            df3_s_t=df3_s_t.set_index('분류')
            tabulator_editors2 = {
                '살생물제품유형': None,
                '제형': None,
                '사용장소': None,
                '적용방식': None,
                '후처리방식': None,
                '대기': None,
                '수계': None,
                '토양': None,
                '대기_사용': None,
                '수계_사용': None,
                '토양_사용': None,
            }
             
            table_b=pn.Column(pn.pane.Markdown("## Ⅰ-3. 물질제조 · 제품생산 및 제품사용단계 배출계수 ("+str(chemical)+")", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(df3_s_t,groups={'물질제조 · 제품생산 배출계수':['대기','수계','토양'],'제품사용단계 배출계수':['대기_사용','수계_사용','토양_사용']},header_align='center',text_align='center',editors=tabulator_editors2,pagination='remote',page_size=5,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))

            stp_removal_df=pd.DataFrame({'CAS-RN':[str(chemical)],'STP 제거율(%)':[float(STP_removal_pecent)]})
            stp_removal_df=stp_removal_df.set_index('CAS-RN')
            tabulator_editors3= {
                'STP 제거율(%)': None,
            }

            table_c=pn.Column(pn.pane.Markdown("## Ⅰ-4. 물질의 STP 제거율 정보", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(stp_removal_df,header_align='center',text_align='center',editors=tabulator_editors3,pagination='remote',page_size=5,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))
            table_c_a=pn.Column(table_c,mark2)


            #%%
            a= df['EF_P_Air']+df['EF_P_Water']+df['EF_P_Soil']
            #매체별 제조단계
            b, c, d = df['EF_P_Air'],  df['EF_P_Water'] , df['EF_P_Soil']
            #인당 일일 
            e = df['Chem_Amount_Total'] * 1e+6/51829053/365
            #총계
            f =df['Chem_Amount_Total']
            #매체별 사용단계
            g ,h , i = f * df['EF_U_Air'] , f* df['EF_U_Water'], f* df['EF_U_Soil']

            #e.g) g[0]: 5836-29-3  , g[1]: 7173-51-5 , g[2] =72963-72-5
            dd2=pd.DataFrame({'구분':['제조 및 생산단계','사용단계','종합'],'전체 환경배출량 (톤/년)':[sum(a),sum(f),sum(a+f)],'대기배출량 (톤/년)':[sum(b),sum(g),sum(b+g)],'수계배출량 (톤/년)':[sum(c),sum(h),sum(c+h)],'토양배출량 (톤/년)':[sum(d),sum(i),sum(d+i)]})
            dd2=dd2.set_index('구분')
            tabulator_editors4 = {
                '전체 환경배출량 (톤/년)': None,
                '대기배출량 (톤/년)': None,
                '수계배출량 (톤/년)': None,
                '토양배출량 (톤/년)': None,
            }
            table=pn.Column(pn.pane.Markdown("### * 단계별 환경배출량", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(dd2,header_align='center',text_align='center',editors=tabulator_editors4,pagination='remote',page_size=5,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))
################################################################
            class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
                def __init__(self): 
                    super(MLPModel, self).__init__()
                    self.linear1 = nn.Linear(6,50)
                    self.actv1 = nn.ReLU()
                    self.linear2 = nn.Linear(50,71)
                    self.actv2 = nn.ReLU()
                    self.linear3 = nn.Linear(71,54)
                    self.actv3 = nn.ReLU()
                    self.linear4 = nn.Linear(54,84)
                    self.actv4 = nn.ReLU()
                    self.linear5 = nn.Linear(84,5)

                def forward(self, x):
                # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
                    x = self.linear1(x)
                    x = self.actv1(x)
                    x = self.linear2(x)
                    x = self.actv2(x)
                    x = self.linear3(x)
                    x = self.actv3(x)
                    x = self.linear4(x)
                    x = self.actv4(x)
                    x = self.linear5(x)
                    return x
            device = torch.device('cpu')   
            model = MLPModel()
            if radio_group2.value == '입력자료 갯수 5개 이하':
                if text_input21.value =='1':
                    a_s=a.loc[chemical]
                    f_s=f.loc[chemical]
                else :
                    a_s=a.loc[chemical]
                    a_s=sum(a_s)
                    f_s=f.loc[chemical]
                    f_s=sum(f_s)
            else:
                if a.loc[chemical].size ==1:
                    a_s=a.loc[chemical]
                    f_s=f.loc[chemical]

                else:
                    a_s=a.loc[chemical]
                    a_s=sum(a_s)
                    f_s=f.loc[chemical]
                    f_s=sum(f_s)
            if select_model.value=='readily biodegradable':
                model.load_state_dict(torch.load('model/sbox_RB.pth', map_location=device))
            elif select_model.value=='inherently biodegradable':
                model.load_state_dict(torch.load('model/sbox_IB.pth', map_location=device))
            elif select_model.value=='not biodegradable':
                model.load_state_dict(torch.load('model/sbox_NB.pth', map_location=device))
            elif select_model.value=='readily biodegradable, failing 10-d window':
                model.load_state_dict(torch.load('model/sbox_RBF.pth', map_location=device))
            data_X=[np.log10(float(a_s+f_s)),float(text_input2.value),float(text_input3.value),np.log10(float(text_input4.value)),np.log10(float(text_input5.value)),np.log10(float(text_input6.value))]
            data_input_x = torch.Tensor(data_X)
            data_reshape_NN = model(data_input_x)
            data_reshape_NN = data_reshape_NN.detach().numpy()
            data_reshape_NN = np.power(10,data_reshape_NN)
#################################################################
            air = data_reshape_NN[0].item()  # j
            water= data_reshape_NN[1].item() # k
            soil = data_reshape_NN[4].item() # l

            domestic_background_conc_data=pd.DataFrame({'CAS_RN':[chemical],'대기 (㎎/㎥)' :[air],'수계 (㎎/L)' :[water],'토양 (㎎/㎏wet)':[soil]})
            domestic_background_conc_data=domestic_background_conc_data.set_index('CAS_RN')
            # domes_table=pn.Column(domestic_background_conc_data,pn.pane.Markdown("<br>"),width=500,height=100)
            mark3=pn.pane.Markdown(" #### ■ 매체별 전국 배경농도 <br>   -  제조 · 생산단계에서 환경으로 배출된 물질량과 <br> 사용단계에서 환경 매체별로 배출된 물질량의 합을 입력자료로 하여 <br> SimpleBox Korea 기반으로 예측 <br><br> ■ 수계 지점별 예측환경농도 <br>   - Tier 1 : 특정 지점에서 배출되는 물질량으로 STP 방류구역의 화학물질 농도를 예측하고, 전국 배경농도와 합산하여 수계 특정 지점의 예측환경농도를 산정 <br><br>   - Tier 2 : 특정 지점에서 배출되는 물질량으로 STP 방류구역의 화학물질 농도를 예측하고, 상류에서 배출된 물질의 누적을 고려한 배경농도와 합산하여 수계 특정 지점의 예측환경농도를 산정", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            tabulator_editors5 = {
                '대기 (㎎/㎥)': None,
                '수계 (㎎/L)': None,
                '토양 (㎎/㎏wet)': None,

            }
            table3=pn.Column(pn.pane.Markdown("## Ⅱ-2. 매체별 전국 배경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(domestic_background_conc_data,header_align='center',text_align='center',editors=tabulator_editors5,pagination='remote',sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))

            e_data=e.to_frame(name='1인 1일 물질사용량 (g/day · 명)')
            e_data=e_data.loc[[chemical]]
            df_e=df_data.copy()
            df_e=df_e[['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식']]
            df_e_t=pd.concat([df_e,e_data],axis=1)
            df_e_t=df_e_t.reset_index(drop=True)
            df_e_t=df_e_t.set_index('분류')
            tabulator_editors6 = {
                '살생물제품유형': None,
                '제형': None,
                '사용장소': None,
                '적용방식': None,
                '후처리방식': None,
                '1인 1일 물질사용량 (g/day · 명)':None,
            }
            
            table4=pn.Column(pn.pane.Markdown("### * 1인 1일 물질사용량 (g/day · 명)", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(df_e_t,header_align='center',text_align='center',editors=tabulator_editors6,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table']))
            # table4=pn.Column(pn.pane.Markdown("### * 1인 1일 물질사용량 (g/day · 명)"),e_data,width=500,height=200,margin=(0,0,50,0))
#############################################

            # index 같은것만 고름
            stp_removal = STP_removal_pecent
##########################################################

############################################################
            # 그래프그릴 데이터 t_df
            line_gdf, stp_point_gdf, t_df, b_pec = rich.take_map_df(chemical,text_input8.value,stp_removal,water,df_e_t)
            stp_point_gdf2 = gpd.GeoDataFrame(pd.merge(line_gdf.loc[:,['RCH_DID', 'MB_NM']], stp_point_gdf, how = 'right', on='RCH_DID'),geometry = 'geometry')
            t_dfs=t_df.copy()
            t_col_list = ['MB_NM',
            '시설명',
            '방류량',
            '인구수',
            '구간체류시간(h)',
            '희석배율',
            'C_local',
            'PEC_local',
            'PEC_WS',]
#            'RCH_LEN',
            t_df = t_df.loc[:,t_col_list]

            ##############
            t_col_list2 = ['MB_NM',
            'RCH_DID',
            'PEC_WS',]
            t_dfs = t_dfs.loc[:,t_col_list2]
            ##############

            t_df_norich=t_df.loc[~t_df.PEC_local.isna(),:]
            t_df_norich.columns =['중권역명','STP명','방류량(㎥/day)','인구수(명)','구간체류시간(h)','희석배율(-)','C_local(㎎/L)','PEC_local(㎎/L)','PEC_WS(㎎/L)']
            t_df__norich=t_df.loc[~t_df.PEC_local.isna(),'PEC_WS']
            t_df_pec=pd.DataFrame({chemical:t_df__norich})
            t_df_pec=t_df_pec.transpose()
            chemical_i_want=chemical
            t_df_2=t_df_norich.set_index('중권역명')

            options2=list(set(t_df_2.index))
            options2.insert(0,'전국')
            select_area=pn.widgets.Select(name='하천 지역선택', options=options2, value='전국', sizing_mode='fixed',margin=(0,1450,20,0),css_classes=['panel-widget'])

            page_2_3_mark=pn.pane.Markdown(" ## Ⅱ-3. 하수처리장별 예측환경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            @pn.depends(xx=select_area.param.value)
            def mock2(xx):
                if xx =='전국':
                    tabulator_editors = {
                        'STP명': None,
                        '방류량(㎥/day)': None,
                        '인구수(명)': None,
                        '구간체류시간(h)': None,
                        '희석배율(-)': None,
                        'C_local(㎎/L)': None,
                        'PEC_local(㎎/L)': None,
                        'PEC_WS(㎎/L)': None,
                    }
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ STP 방류수역 예측환경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(t_df_2,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,45,0),css_classes=['table']))
                else: 
                    t_df_2_t_data=t_df_2.loc[[xx]]
                    tabulator_editors = {
                        'STP명': None,
                        '방류량(㎥/day)': None,
                        '인구수(명)': None,
                        '구간체류시간(h)': None,
                        '희석배율(-)': None,
                        'C_local(㎎/L)': None,
                        'PEC_local(㎎/L)': None,
                        'PEC_WS(㎎/L)': None,
                    }
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ STP 방류수역 예측환경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(t_df_2_t_data,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,45,0),css_classes=['table']))                    
                return table_n

            t_dfs.columns =['중권역명','RCH_DID','PEC_WS(㎎/L)',]
            t_dfs=t_dfs.set_index('중권역명')
            @pn.depends(xx=select_area.param.value)
            def mock4(xx):
                if xx =='전국':
                    tabulator_editors = {
                        'RCH_DID': None,
                        'PEC_WS(㎎/L)': None,
                    }
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ 표준유역 예측환경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(t_dfs,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=10,sizing_mode='fixed',margin=(0,0,45,0),css_classes=['table']))
                else: 
                    t_dfs_t=t_dfs.loc[[xx]]
                    tabulator_editors = {
                        'RCH_DID': None,
                        'PEC_WS(㎎/L)': None,
                    }
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ 표준유역 예측환경농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(t_dfs_t,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=10,sizing_mode='fixed',margin=(0,0,45,0),css_classes=['table']))
                return table_n


            @pn.depends(xx=select_area.param.value)
            def pec_df(xx):
                t_df_sort=t_df.copy()
                t_df_sort.columns =['중권역명','STP명','방류량','인구수','구간체류시간(h)','희석배율','C_local','PEC_local','PEC_WS']
                t_df_sort=t_df_sort.set_index('중권역명')
                t_df_sort=t_df_sort.loc[[xx]]
                t_df_sort=t_df_sort.reset_index()
                return t_df_sort

            text_input9 = pn.widgets.TextInput(name='예측무영향농도(PNEC) (㎍/L)', placeholder='PNEC값 입력',sizing_mode='fixed',width=160,margin=(0,100,10,10),css_classes=['panel-widget'])
            @pn.depends(xx=select_area.param.value,yy=text_input9.param.value)
            def pec_fig(xx,yy):
                if xx =='전국':
                    t_df_plot_rich=t_df.loc[:,'PEC_WS'].sort_values()
                    t_df_plot_norich=t_df.loc[~t_df.PEC_local.isna(),'PEC_WS'].sort_values()
                    t_df_plot_rich= t_df_plot_rich.to_list()
                    t_df_plot_norich=t_df_plot_norich.to_list()
                else:
                    t_df_sort=pec_df(xx)
                    t_df_plot_rich=t_df_sort.loc[:,'PEC_WS'].sort_values()
                    t_df_plot_norich=t_df_sort.loc[~t_df_sort.PEC_local.isna(),'PEC_WS'].sort_values()
                    t_df_plot_rich= t_df_plot_rich.to_list()
                    t_df_plot_norich=t_df_plot_norich.to_list()
                y = np.arange(0, len(t_df_plot_rich))/len(t_df_plot_rich)
                y1= np.arange(0, len(t_df_plot_norich))/len(t_df_plot_norich)
                fig = go.Figure()
                fig.add_trace(go.Scatter(mode="markers",x=t_df_plot_norich, y=y1,marker_symbol="triangle-up",name='STP 방류수역'))
                fig.add_trace(go.Scatter(mode="markers",x=t_df_plot_rich, y=y,marker_symbol="triangle-up",name='표준유역'))
                # fig.update_xaxes(title='Concentration (μg/L)',type="log",title_font_family="NanumBarunGothic") #(㎎/L)
                fig.update_xaxes(title='Concentration(㎎/L)',type="log",title_font_family="NanumBarunGothic") #(㎎/L)                
                fig.update_yaxes(title='Fraction of local',title_font_family="NanumBarunGothic")
                # fig.update_layout(title=chemical_i_want+"의 수계 지점별 예측환경농도(PEC) 분포",title_font_family="NanumBarunGothic")
                fig.add_vline(x=b_pec, line_width=2, line_dash="dash", line_color="red")
                if yy=='':
                    None
                else:
                    fig.add_vline(x=float(yy), line_width=2, line_dash="dash", line_color="green")
                fig.add_hrect(y0=0.9, y1=0.95, line_width=2, line_color="red",fillcolor="red", opacity=0.2)
                pec_fig_mark=pn.pane.Markdown("### ■ "+chemical_i_want+"의 수계 지점별 예측환경농도(PEC) 분포 <br> ("+str(xx)+")", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
                return pn.Column(pec_fig_mark,fig, width=650, height=500,margin=(20,100,20,0))           

            
            
            button_t = pn.widgets.Button(name='입력', button_type='primary',sizing_mode='fixed',width=140,css_classes=['widget-button'])
            output1 = pn.widgets.TextInput(name='표준유역 중 PNEC 초과비율(%)',value='',disabled=True,sizing_mode='fixed',width=160,css_classes=['panel-widget'])
            output2 = pn.widgets.TextInput(name='STP 방류수역 중 PNEC 초과비율(%)',value='',disabled=True,sizing_mode='fixed',width=160,css_classes=['panel-widget'])

            #표준유역
            def calculate_ratio():
                if select_area.value =='전국':
                    df_test=t_df
                    x=df_test.loc[:,'PEC_WS'].sort_values()
                    x2 = sum(x > float(text_input9.value))
                    x3=x2/len(x)*100
                    x3=round(x3,4)
                else:
                    df_test=pec_df(select_area.value)
                    x=df_test.loc[:,'PEC_WS'].sort_values()
                    x2 = sum(x > float(text_input9.value))
                    x3=x2/len(x)*100
                    x3=round(x3,4)
                return x3

            #STP 방류수역
            def calculate_ratio2():
                if select_area.value =='전국':
                    df_test=t_df
                    x=df_test.loc[~df_test.PEC_local.isna(),'PEC_WS'].sort_values()
                    x2 = sum(x > float(text_input9.value))
                    x3=x2/len(x)*100
                    x3=round(x3,4)
                else:
                    df_test=pec_df(select_area.value)
                    x=df_test.loc[~df_test.PEC_local.isna(),'PEC_WS'].sort_values()
                    x2 = sum(x > float(text_input9.value))
                    x3=x2/len(x)*100
                    x3=round(x3,4)                     
                return x3

            def pnec_value():
                if text_input9.value=='':
                    x=''
                else:
                    x=text_input9.value
                return x

            def button_event_t(event):
                output1.value = str(calculate_ratio())
                output2.value = str(calculate_ratio2())
                text_input9.value = str(pnec_value())          
            button_t.on_click(button_event_t)

            pnec_ratio=pn.Column(pn.pane.Markdown("#### ■ 예측무영향농도(PNEC) (㎍/L) 입력비교", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.Row(text_input9,button_t),output1,output2)

            def pec_table():
                t_df_rich=t_df.loc[:,'PEC_WS'].sort_values()
                t_df_rich=t_df_rich.to_list()
                t_df_norich=t_df.loc[~t_df.PEC_local.isna(),'PEC_WS'].sort_values()
                t_df_norich=t_df_norich.to_list()
                t_df_90_r=t_df_rich[int(len(t_df_rich)*0.9-1)]
                t_df_95_r=t_df_rich[int(round(len(t_df_rich)*0.95)-1)]
                t_df_90_nr=t_df_norich[int(len(t_df_norich)*0.9-1)]
                t_df_95_nr=t_df_norich[int(len(t_df_norich)*0.95-1)]
                table_pec=pd.DataFrame({'90%':[t_df_90_r,t_df_90_nr],'95%':[t_df_95_r,t_df_95_nr]},index=['표준유역 PEC_WS(㎎/L)','STP 방류수역PEC_WS(㎎/L)'])
                tabulator_editors = {
                    '90%': None,
                    '95%': None,
                }                
                return pn.Column(pn.pane.Markdown("#### ■ 예측환경농도(PEC)의 90% & 95% 값", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.widgets.Tabulator(table_pec,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',sizing_mode='fixed',margin=(10,0,50,0),css_classes=['table']))
######################################### 3페이지 연산 ##################################################################################################################################

            material_conc,intake_db,intake_db_A_idx,intake_db_B_idx,intake_db_C_idx,intake_db_D_idx,intake_db_E_idx,T_if2,T_if3,T_if4,Exp_mean,\
            Exp2_A_mean,Exp3_A_mean,Exp4_A_mean,Expw_A_mean,Exp_A_mean,\
            Exp2_B_mean,Exp3_B_mean,Exp4_B_mean,Expw_B_mean,Exp_B_mean,\
            Exp2_C_mean,Exp3_C_mean,Exp4_C_mean,Expw_C_mean,Exp_C_mean,\
            Exp2_D_mean,Exp3_D_mean,Exp4_D_mean,Expw_D_mean,Exp_D_mean,\
            Exp2_E_mean,Exp3_E_mean,Exp4_E_mean,Expw_E_mean,Exp_E_mean=aqua.user_input(chemical_i_want,t_df_pec,t_df_2,stp_removal,df,text_input4.value)

            options3=list(set(material_conc.index))
            options3.insert(0,'전국')
            select_area2=pn.widgets.Select(name='하천 지역선택', options=options2, value='전국', sizing_mode='fixed',margin=(0,1450,20,0),css_classes=['panel-widget'])

            @pn.depends(xx=select_area2.param.value)
            def mock3(xx):
                if xx =='전국':
                    tabulator_editors = {
                        'STP명': None,
                        '음용수(µg/L)': None,
                        '영양단계2(µg/g)': None,
                        '영양단계3(µg/g)': None,
                        '영양단계4(µg/g)': None,
                    }
                    table_n=pn.Column(select_area2,pn.widgets.Tabulator(material_conc,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table'])) 
                else: 
                    material_conc_data=material_conc.loc[[xx]]
                    tabulator_editors = {
                        'STP명': None,
                        '음용수(µg/L)': None,
                        '영양단계2(µg/g)': None,
                        '영양단계3(µg/g)': None,
                        '영양단계4(µg/g)': None,
                    }
                    table_n=pn.Column(select_area2,pn.widgets.Tabulator(material_conc_data,header_align='center',text_align='center',editors=tabulator_editors,pagination='remote',page_size=8,sizing_mode='fixed',margin=(0,0,95,0),css_classes=['table'])) 
                return table_n

            figure_4=pn.Column(pn.pane.Markdown("## Ⅲ-2. 음용수 및 어패류 중 물질농도", style={'font-family': 'NanumBarunGothic','font-size':'20px'}))
            # table_w=pn.Column(table_n,width=500,height=200,margin=(0, 50, 20, 0))
            # table_w_a=pn.Column(pn.pane.Markdown("## Ⅲ-2. 음용수 및 어패류 중 물질농도"),figure_4,table_w)

            age1='1세'
            age2='2~3세'
            age3='4~6세'
            age4='7~12세'
            age5='청소년 및 성인'
            age6='노출기여도'

            # # # 체중
            # def weight_plot(x,y):
            #     x1=list(x)
            #     hist_data = [x1]
            #     group_labels = [y]
            #     colors = ['#333F44']

            #     fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
            #     fig.update_xaxes(title="체중 (㎏)",title_font_family="NanumBarunGothic")
            #     fig.update_yaxes(title="빈도 (-)",title_font_family="NanumBarunGothic")
            #     fig.update_layout(title_text=y,title_font_family="NanumBarunGothic",width=350,height=350,showlegend=False)
            #     return fig

            # def weight_plot_pn():
            #     fig1= weight_plot(intake_db.iloc[intake_db_A_idx,:]['HE_wt'],age1)
            #     fig2= weight_plot(intake_db.iloc[intake_db_B_idx,:]['HE_wt'],age2)
            #     fig3= weight_plot(intake_db.iloc[intake_db_C_idx,:]['HE_wt'],age3)
            #     fig4= weight_plot(intake_db.iloc[intake_db_D_idx,:]['HE_wt'],age4)
            #     fig5= weight_plot(intake_db.iloc[intake_db_E_idx,:]['HE_wt'],age5)  
            #     return pn.Column(pn.pane.Markdown("### 체중", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.Row(fig1,fig2,fig3,fig4,fig5))
            # # 체중
            def weight_plot(x,x2,x3,x4,x5,y1,y2,y3,y4,y5):
                x_1=list(x)
                x_2=list(x2)
                x_3=list(x3)
                x_4=list(x4)
                x_5=list(x5)
                hist_data = [x_1,x_2,x_3,x_4,x_5]
                group_labels = [y1,y2,y3,y4,y5]

                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
                fig.update_xaxes(title="체중 (㎏)",title_font_family="NanumBarunGothic")
                fig.update_yaxes(title="빈도 (-)",title_font_family="NanumBarunGothic")
                fig.update_layout(width=550,height=550,showlegend=True)
                return fig

            def weight_plot_pn():
                fig1= weight_plot(intake_db.iloc[intake_db_A_idx,:]['HE_wt'],intake_db.iloc[intake_db_B_idx,:]['HE_wt'],intake_db.iloc[intake_db_C_idx,:]['HE_wt'],intake_db.iloc[intake_db_D_idx,:]['HE_wt'],intake_db.iloc[intake_db_E_idx,:]['HE_wt'],age1,age2,age3,age4,age5)
                return pn.Column(pn.pane.Markdown("### ■ 체중", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),fig1)

            # # 음용수
            def dis_water_plot(x,x2,x3,x4,x5,y1,y2,y3,y4,y5):
                x_1=list(x)
                x_2=list(x2)
                x_3=list(x3)
                x_4=list(x4)
                x_5=list(x5)
                hist_data = [x_1,x_2,x_3,x_4,x_5]
                group_labels = [y1,y2,y3,y4,y5]
                # colors = ['#333F44']

                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
                fig.update_xaxes(title="음용수 섭취량 (L/day)",title_font_family="NanumBarunGothic")
                fig.update_yaxes(title="빈도 (-)",title_font_family="NanumBarunGothic")
                fig.update_layout(width=550,height=550,showlegend=True)
                return fig

            def dis_water_plot_pn():
                fig1= dis_water_plot(intake_db.iloc[intake_db_A_idx,:]['intake.w_liter.per.day'],intake_db.iloc[intake_db_B_idx,:]['intake.w_liter.per.day'],intake_db.iloc[intake_db_C_idx,:]['intake.w_liter.per.day'],intake_db.iloc[intake_db_D_idx,:]['intake.w_liter.per.day'],intake_db.iloc[intake_db_E_idx,:]['intake.w_liter.per.day'],age1,age2,age3,age4,age5)
                return pn.Column(pn.pane.Markdown("### ■ 음용수 섭취량", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),fig1)

            def dist_fish_list(x):
                x=pd.Series(x)
                x=list(x)
                return x

            def dis_fish_plot(x,x1,x2,y):
                test=pd.DataFrame({"영양단계2":x,"영양단계3":x1,"영양단계4":x2})
                fig=test.hvplot.kde(title=y,xlabel="어패류 섭취량 (g/day)",ylabel="빈도 (-)",xlim=(0,5),width=350,height=350).opts(legend_position='top_right')
                return fig

            def dis_fish_plot_pn():
                fig1= dis_fish_plot(dist_fish_list(T_if2[intake_db_A_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_A_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_A_idx,:].sum(axis=1)),age1)
                fig2= dis_fish_plot(dist_fish_list(T_if2[intake_db_B_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_B_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_B_idx,:].sum(axis=1)),age2)
                fig3= dis_fish_plot(dist_fish_list(T_if2[intake_db_C_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_C_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_C_idx,:].sum(axis=1)),age3)
                fig4= dis_fish_plot(dist_fish_list(T_if2[intake_db_D_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_D_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_D_idx,:].sum(axis=1)),age4)
                fig5= dis_fish_plot(dist_fish_list(T_if2[intake_db_E_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_E_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_E_idx,:].sum(axis=1)),age5)
                return pn.Column(pn.pane.Markdown("### ■ 어패류 섭취량", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.Row(fig1,fig2,fig3,fig4,fig5,sizing_mode='fixed'),mark2)       

            # # In[] 누적분포함수

            mark9=pn.pane.Markdown("## Ⅲ-3. 연령별 인구집단의 노출계수 분포",  style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            mark10=pn.pane.Markdown("## Ⅳ-1. 연령별 수계 유래 간접 인체노출량 plot", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            mark11=pn.pane.Markdown("## Ⅳ-2. 연령별 수계 유래 간접 인체노출량 누적 bar plot", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            mark12=pn.pane.Markdown("## Ⅳ-3. 연령별 음용수/어패류 섭취 노출기여도 pie chart", style={'font-family': 'NanumBarunGothic','font-size':'20px'})

            def cdf_plot_fish(a,b,c,f):
                sorted_Exp2_mean = np.sort(a) #Exp2_A_mean
                sorted_Exp3_mean = np.sort(b) #Exp3_A_mean
                sorted_Exp4_mean = np.sort(c) #Exp4_A_mean
                sum=sorted_Exp2_mean + sorted_Exp3_mean+ sorted_Exp4_mean
                p_sum = 1. * np.arange(len(sum)) / float(len(sum) - 1)

                # sorted_Exp_mean = np.sort(e)
                # p_ = 1. * np.arange(len(sorted_Exp_mean)) / float(len(sorted_Exp_mean) - 1)

                fig2 = go.Figure()
                fig2.add_scatter(x=sum, y=p_sum,name='어패류')
                # fig2.add_scatter(x=np.log10(sorted_Expw_mean+1), y=p_w,name='음용수')
                # fig2.add_scatter(x=np.log10(sorted_Exp_mean+1), y=p_,name='전체')
                fig2.update_xaxes(title="노출량 (μg/kg/Day)",type="log")
                fig2.update_yaxes(range=[0.98, 1],fixedrange=True,title="누적빈도")
                fig2.update_layout(width=400,height=350,title=f,title_font_family="NanumBarunGothic")
                return fig2
            
            def cdf_plot_w(d,f):
                sorted_Expw_mean = np.sort(d)
                p_w = 1. * np.arange(len(sorted_Expw_mean)) / float(len(sorted_Expw_mean) - 1)
                fig = go.Figure()
                fig.add_scatter(x=sorted_Expw_mean, y=p_w,name='음용수')
                fig.update_xaxes(title="노출량 (μg/kg/Day)",title_font_family="NanumBarunGothic",type="log")
                fig.update_yaxes(title="누적빈도",title_font_family="NanumBarunGothic")
                fig.update_layout(width=400,height=350,title=f,title_font_family="NanumBarunGothic")
                return fig


            def cdf_case():
                m_idx = list(intake_db[intake_db['sex.x'] == 1].index)
                w_idx = list(intake_db[intake_db['sex.x'] == 2].index)

                m_sorted_Exp_mean = np.sort(Exp_mean[m_idx])
                p_m = 1. * np.arange(len(m_sorted_Exp_mean)) / float(len(m_sorted_Exp_mean) - 1)
                w_sorted_Exp_mean = np.sort(Exp_mean[w_idx])
                p_w = 1. * np.arange(len(w_sorted_Exp_mean)) / float(len(w_sorted_Exp_mean) - 1)
                sorted_Exp_mean = np.sort(Exp_mean)
                p = 1. * np.arange(len(sorted_Exp_mean)) / float(len(sorted_Exp_mean) - 1)

                # In[]
                fig = go.Figure()
                fig.add_scatter(x=sorted_Exp_mean, y=p,name='전체')
                fig.add_scatter(x=m_sorted_Exp_mean, y=p_m,name='남성')
                fig.add_scatter(x=w_sorted_Exp_mean, y=p_w,name='여성')
                fig.update_xaxes(title="노출량 (μg/kg/Day)",title_font_family="NanumBarunGothic",type="log")
                fig.update_yaxes(title="누적빈도",title_font_family="NanumBarunGothic")
                fig.update_layout(width=400,height=350,title="어패류 및 음용수 통합 노출량",title_font_family="NanumBarunGothic")
                return fig

            def cdf_plot_all(e,f):
                sorted_Exp_mean = np.sort(e)
                p_ = 1. * np.arange(len(sorted_Exp_mean)) / float(len(sorted_Exp_mean) - 1)
                fig = go.Figure()
                fig.add_scatter(x=sorted_Exp_mean, y=p_,name='전체')
                fig.update_xaxes(title="노출량 (μg/kg/Day)",title_font_family="NanumBarunGothic",type="log")
                fig.update_yaxes(title="누적빈도",title_font_family="NanumBarunGothic")
                fig.update_layout(width=400,height=350,title=f,title_font_family="NanumBarunGothic")
                return fig

            def cdf_plot_page():
                figall=cdf_case()
                pieall=pie_all()
                figA=cdf_plot_fish(Exp2_A_mean,Exp3_A_mean,Exp4_A_mean,age1)
                figB=cdf_plot_fish(Exp2_B_mean,Exp3_B_mean,Exp4_B_mean,age2)
                figC=cdf_plot_fish(Exp2_C_mean,Exp3_C_mean,Exp4_C_mean,age3)
                figD=cdf_plot_fish(Exp2_D_mean,Exp3_D_mean,Exp4_D_mean,age4)
                figE=cdf_plot_fish(Exp2_E_mean,Exp3_E_mean,Exp4_E_mean,age5)

                figF=cdf_plot_w(Expw_A_mean,age1)
                figG=cdf_plot_w(Expw_B_mean,age2)
                figH=cdf_plot_w(Expw_C_mean,age3)
                figI=cdf_plot_w(Expw_D_mean,age4)
                figJ=cdf_plot_w(Expw_E_mean,age5)

                figK=cdf_plot_all(Exp_A_mean,age1)
                figL=cdf_plot_all(Exp_B_mean,age2)
                figM=cdf_plot_all(Exp_C_mean,age3)
                figN=cdf_plot_all(Exp_D_mean,age4)
                figO=cdf_plot_all(Exp_E_mean,age5)
                mark_f=pn.pane.Markdown("### 어패류 섭취 노출량", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
                mark_w=pn.pane.Markdown("### 음용수 섭취 노출량", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
                mark_a=pn.pane.Markdown("### 전체(어패류 + 음용수)", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
                return pn.Column(pn.Row(figall,pieall),mark2,mark_f,pn.Row(figA,figB,figC,figD,figE),mark2,mark_w,pn.Row(figF,figG,figH,figI,figJ),mark2,mark_a,pn.Row(figK,figL,figM,figN,figO))      #

            def pie_chart(a,b,c,d,e):
                labels = ['음용수', '영양단계2', '영양단계3', '영양단계4']
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                portion_list = np.array([a.sum(), b.sum(), c.sum(), d.sum()])
                fig = go.Figure(data=[go.Pie(labels=labels, values=portion_list, hole=.5)])
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=350,height=350,title=e,title_font_family="NanumBarunGothic")
                return fig

            def pie_all():
                fig=pie_chart(Expw_E_mean,Exp2_E_mean,Exp3_E_mean,Exp4_E_mean,age6)
                return fig

            def pie_page():
                figA=pie_chart(Expw_A_mean,Exp2_A_mean,Exp3_A_mean,Exp4_A_mean,age1)
                figB=pie_chart(Expw_B_mean,Exp2_B_mean,Exp3_B_mean,Exp4_B_mean,age2)
                figC=pie_chart(Expw_C_mean,Exp2_C_mean,Exp3_C_mean,Exp4_C_mean,age3)
                figD=pie_chart(Expw_D_mean,Exp2_D_mean,Exp3_D_mean,Exp4_D_mean,age4)
                figE=pie_chart(Expw_E_mean,Exp2_E_mean,Exp3_E_mean,Exp4_E_mean,age5)
                return pn.Row(figA,figB,figC,figD,figE)

            def stack_bar():
                age=['1세', '2~3세', '4~6세', '7~12세','청소년 및 성인']
                # colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                fig = go.Figure(data=[
                    # go.Bar(name='전체', x=age, y=[Exp_A_mean.sum(),Exp_B_mean.sum(),Exp_C_mean.sum(),Exp_D_mean.sum(),Exp_E_mean.sum()]),
                    go.Bar(name='음용수', x=age, y=[Expw_A_mean.sum(),Expw_B_mean.sum(),Expw_C_mean.sum(),Expw_D_mean.sum(),Expw_E_mean.sum()]),
                    go.Bar(name='영양단계2', x=age, y=[Exp2_A_mean.sum(),Exp2_B_mean.sum(),Exp2_C_mean.sum(),Exp2_D_mean.sum(),Exp2_E_mean.sum()]),
                    go.Bar(name='영양단계3', x=age, y=[Exp3_A_mean.sum(),Exp3_B_mean.sum(),Exp3_C_mean.sum(),Exp3_D_mean.sum(),Exp3_E_mean.sum()]),
                    go.Bar(name='영양단계4', x=age, y=[Exp4_A_mean.sum(),Exp4_B_mean.sum(),Exp4_C_mean.sum(),Exp4_D_mean.sum(),Exp4_E_mean.sum()]),
                ])
                # Change the bar mode
                # fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=800,height=400,barmode='stack')
                fig.update_yaxes(title="노출량 (μg/kg/Day)",title_font_family="NanumBarunGothic",type="log")
                return fig

            mark5=pn.pane.Markdown("## Ⅱ-1. 단계별 환경배출량", style={'font-family': 'NanumBarunGothic','font-size':'20px'})
            flow_3=pn.Column(pn.pane.Markdown("## Ⅲ-1 수환경 인체 간접노출량 산정방법", style={'font-family': 'NanumBarunGothic','font-size':'20px'}),pn.pane.JPG('FF_exp.jpg',height=470,width=800,margin=(0,0,50,0)))

            @pn.depends(xx=select_area.param.value)
            def map_s(xx):
                def get_geo_map(MB: str, stp_geoinfo_gdf: gpd.GeoDataFrame, line_geoinfo_gdf: gpd.GeoDataFrame):
                    
                    if MB == '전국':
                        pass
                    else:
                        stp_geoinfo_gdf = stp_geoinfo_gdf.loc[stp_geoinfo_gdf.MB_NM ==MB]
                        line_geoinfo_gdf = line_geoinfo_gdf.loc[line_geoinfo_gdf.MB_NM ==MB]
                    ## 지도그리는 함수부분 ##
                    color_list_1 = stp_geoinfo_gdf.PEC_local.values.tolist() + line_geoinfo_gdf.PEC_WS.tolist()
                    color_list_1.sort()

                    # index_list = [0.000002,0.000004,0.0000045,0.00000468,0.00000481,0.00000513,0.00000569,0.00000704,0.00000809,0.0000111,0.0000196,]
                    index_list = [0.000001,0.000002,0.0000021,0.0000022,0.0000023,0.0000024]
                    col_bar_list = []


                    r_start = 255
                    g_start = 182
                    b_start = 94

                    r_end = 0
                    g_end = 0
                    b_end = 0




                    for i in range(len(index_list)):

                        r = r_start - (r_start - r_end)*(i/len(index_list))
                        g = g_start - (g_start - g_end)*(i/len(index_list))
                        b = b_start - (b_start - b_end)*(i/len(index_list))
                    
                        col_bar_list.append((r,g,b))


                    con_step = cmp.LinearColormap(
                        colors=col_bar_list,
                        # index=index_list,
                        vmin=np.min(color_list_1),
                        vmax=np.max(color_list_1),
                        caption='PEC(mg/L)'
                    )

                    m = line_geoinfo_gdf.explore(tiles = "CartoDB positron", column = 'PEC_WS', legend = True, cmap = con_step,)
                    stp_geoinfo_gdf.explore(m=m, column = 'PEC_local', cmap = con_step, marker_type = folium.Circle(radius=300, fill = 'black'),)
                    return m

                map_1 = get_geo_map(xx,stp_point_gdf2, line_gdf)
                pp=pn.panel(map_1,sizing_mode="fixed",width=900,height=450,margin=(10,0,20,-400))
                return pn.Column(pn.pane.Markdown(" ### ■ 예측농도 지도("+str(xx)+")" ,style={'font-family': 'NanumBarunGothic','font-size':'20px'},margin=(0,0,0,-400)),pp)

            def pie_part():
                colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                dd3=dd2.drop(labels='종합',axis=0)
                labels = list(dd3.index)
                portion_list = list(dd3['전체 환경배출량 (톤/년)'])
                portion_list2 = list(dd3['대기배출량 (톤/년)'])
                portion_list3 = list(dd3['수계배출량 (톤/년)'])
                portion_list4 = list(dd3['토양배출량 (톤/년)'])
                fig = go.Figure(data=[go.Pie(labels=labels, values=portion_list, hole=.5)])
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(width=375,height=375)
                fig.update_layout(title='전체 환경배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig2 = go.Figure(data=[go.Pie(labels=labels, values=portion_list2, hole=.5)])
                fig2.update_traces(marker=dict(colors=colors))
                fig2.update_layout(width=375,height=375)
                fig2.update_layout(title='대기배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig3 = go.Figure(data=[go.Pie(labels=labels, values=portion_list3, hole=.5)])
                fig3.update_traces(marker=dict(colors=colors))
                fig3.update_layout(width=375,height=375)
                fig3.update_layout(title='수계배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig4 = go.Figure(data=[go.Pie(labels=labels, values=portion_list4, hole=.5)])
                fig4.update_traces(marker=dict(colors=colors))
                fig4.update_layout(width=375,height=375)
                fig4.update_layout(title='토양배출량 (톤/년)',title_font_family="NanumBarunGothic")
                return pn.Column(pn.Row(fig,fig2),pn.Row(fig3,fig4),sizing_mode="fixed",margin=(10,10,10,10))
                # return pn.Row(fig,fig2,fig3,fig4,sizing_mode="fixed",margin=(10,10,10,10))

            ###### 수식 마크다운들
###### 1 페이지
            title_1 = pn.pane.Markdown("""
            -------------------                           
            * **STP 방류수 농도 산정식**
            <br>
            """,sizing_mode="fixed", style={'font-family': 'NanumBarunGothic','font-size':'25px'},margin=(15,0,20,15),width=650)

            a = pn.pane.LaTeX(r"""
            $\begin{aligned} & {C_{local}}_{eff} = \frac {{E_{local}}_{water} \times 10^6 \times (1-STP_{removal} \times 100)}{EFFLUENT_{stp}} & \end{aligned} $
            """, style={'font-size': '25px'},sizing_mode="fixed", margin=(15,0,20,15),width=650)

            defi_1 = pn.pane.Markdown("""
            용어: 
            <blockquote>
            <p> $${C_{local}}_{eff}$$ (㎎/ℓ) : STP 방류수 중 물질농도</p>
            <p> $${E_{local}}_{water}$$ (㎏/day) : 하수로 배출되는 물질량 (=원단위 x 인구수)</p>
            <p> $$STP_{removal}$$ (%) : STP 처리과정에서의 물질제거율</p>
            <p> $$EFFLUENT_{stp}$$ (ℓ/day) : STP 일일 방류량</p>
            </blockquote>
            """, sizing_mode='fixed', style={'font-family': 'NanumBarunGothic','font-size':'20px'}, margin=(15,0,20,15),width=650)

            page_1=pn.Column(title_1,a,defi_1,margin=(0,0,0,0))

            title_2 = pn.pane.Markdown("""
                -------------------                           
                * **STP 방류지점의 예측환경농도 산정식**
                """, sizing_mode='fixed', style={'font-family': 'NanumBarunGothic','font-size':'25px'}, margin=(15,0,20,15),width=650)

            b = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & {PEC_{local}}_{water} = \frac {{C_{local}}_{eff}}{(1+{F_{oc}}_{susp} \cdot K_{oc}\cdot SUSP_{water} \cdot 10^{-6} \cdot DILUTION)} + PEC_{regional} & \end{aligned}$
            """, style={'font-size': '25px'},sizing_mode="fixed", margin=(15,0,20,15),width=650)

            defi_2 = pn.pane.Markdown("""
            용어: 
            <blockquote>
            <p> $${PEC_{local}}_{water}$$ (㎎/ℓ) : STP 방류수역의 수계 예측환경농도</p>
            <p> $${C_{local}}_{eff}$$ (㎎/ℓ) : STP 방류수 중 물질농도</p>
            <p> $${F_{oc}}_{susp}$$ (-) : 부유물질 중 유기탄소 비율</p>
            <p> $$K_{oc}$$ (ℓ/㎏) : 물질의 유기탄소-물 분배계수</p>
            <p> $$SUSP_{water}$$ (㎎/ℓ) : 하천 부유물질 농도</p>
            <p> $$DILUTION$$ (-) : 희석배율</p>
            <p> $$PEC_{regional}$$ (㎎/ℓ) : 전국단위 수계 배경농도</p>
            </blockquote>
            """, sizing_mode='fixed', style={'font-family': 'NanumBarunGothic','font-size':'20px'}, margin=(15,0,20,15),width=650)

            page_1_2=pn.Column(title_2,b,defi_2,margin=(0,0,0,10))

            fomu1=pn.Row(page_1,page_1_2)
####### 3페이지 
            title_3 = pn.pane.Markdown("""
            -------------------                           
            * **수환경 인체 간접노출량 산정식**
            """,style={'font-size': '25px'}, width=500)

            c = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & Exposure \text { } (ug/kg \cdot day) = \frac {Ingestion \text { } Amount \text { } (g/day \text { or } L/day) \times Conc. \text { } (ug/g \text { or }ug/L)}{BodyWeight\text { } (kg)} & \end{aligned}$
            """, style={'font-size': '20px'})

            fomu2=pn.Column(title_3,c)
########3-2 page
            title_4 = pn.pane.Markdown("""
            -------------------                           
            * **어패류 중 물질농도 계산식**
            """,style={'font-size': '25px'}, width=500)

            d = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & BAF = \frac {C_B}{C_W}  = (1-L_B)+ \frac {k_1 \cdot \phi + (k_D \cdot \beta \cdot \tau \cdot \phi \cdot L_D \cdot K_{ow})} {k_2 + k_E + k_G + k_M} & \end{aligned}$
            """, style={'font-size': '25px'})

            e = pn.pane.LaTeX(r"""                  
            $\begin{aligned} \phi = \frac {1}{1+{\chi}_{POC} \cdot 0.35 \cdot K_{ow} + {\chi}_{DOC} \cdot 1 \cdot 0.35 \cdot K_{ow}} \end{aligned}$
            """, style={'font-size': '25px'})

            f = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_1 = \frac {1}{(0.01+ \frac{1}{K_{ow}})\cdot W^{0.4}}  \end{aligned}$
                    
            """, style={'font-size': '25px'})

            g = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_2 = \frac {k_1}{L_B \cdot K_{ow}}  \end{aligned}$
                    
            """, style={'font-size': '25px'})

            h = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_D = \frac {0.02 \cdot W^{-0.15} \cdot e^{(0.06 \cdot T)}}{5.1 \cdot 10^{-8} \cdot K_{ow} +2}  \end{aligned}$
                    
            """, style={'font-size': '25px'})

            defi_3 = pn.pane.Markdown("""
            ------------------- 
            용어: 
            <blockquote>
            <p> $$L_B$$ : Lipid content of organism (20%) </p>
            <p> $${\chi}_{POC}$$ : Concentration of particulate organic carbon ( 5 x 10<sup>-7</sup> g/ml ) </p>
            <p> $${\chi}_{DOC}$$ : Concentration of dissolved organic carbon ( 5 x 10<sup>-7</sup>  g/ml) </p>
            <p> β : Overall food web biomagnification factor (1)</p>
            <p> τ : Maximum trophic dilution factor (130)</p>
            <p> $$L_D$$ : Lipid content of lowest trophic level organisms (1%) </p>
            <p> $$K_{ow}$$ : Octanol-water partition coefficient (Chemical dependent) </p>
            <p> $$ T $$ : Mean  water  temperature (10℃) </p>
            <p> $$ k_E $$ : Fecal egestion rate constant (0.125 $$\cdot k_D $$) </p>
            <p> $$ k_G $$ : Growth rate constant (0.0005 $$\cdot W^{0.2} $$) </p>
            <p> $$ k_M $$ : Metabolic transformation rate constant (0 day<sup>-1</sup>(default)) </p>
            <p> W : Weight of organism (별도 제시)</p>

            """, sizing_mode='fixed', style={'font-family': 'NanumBarunGothic','font-size':'20px'}, margin=(15,0,20,15),width=750)

            defi_4 = pn.pane.Markdown("""
            ------------------- 
            영양단계별 Organism weight (g): 

            <blockquote>
            <p> FRC1 :&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;FRC2 :&emsp;&emsp;&emsp;&emsp;&emsp;FRC3 :   </p>
            <p>   &nbsp; &nbsp;우렁: 0.003&emsp;&emsp;&emsp;&emsp;미꾸리: 0.01&emsp;&emsp;&emsp;송어: 1</p>
            <p>   &nbsp; &nbsp;다슬기: 0.003&emsp;&emsp;&emsp;피라미: 0.01&emsp;&emsp;&emsp;메기: 1</p>
            <p>   &nbsp; &nbsp;재첩: 0.003&emsp;&emsp;&emsp;&emsp;붕어: 1</p>
            <p>   &nbsp; &nbsp;빙어: 0.01&ensp;&emsp;&emsp;&emsp;&emsp;향어: 5</p>
            <p>   &emsp;&emsp;&emsp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;잉어: 3</p>
            <p>   &emsp;&emsp;&emsp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;동자개: 0.5</p>

            </blockquote>
            """, sizing_mode='fixed', style={'font-family': 'NanumBarunGothic','font-size':'20px'}, margin=(15,0,20,15),width=550)
            # fomu3_sub=pn.Row(defi_3,defi_4)
            fomu3=pn.Column(title_4,d,e,f,g,pn.Row(defi_3,defi_4))
#######################################
            def get_stream_pec_plot(site: str, line_gdf_local, stp_point_gdf_local):

                xx = site
                rch_list = []
                rch_len_list = []
                stp_list = []

                if xx != '전국':
                    mb_gdf = line_gdf_local.loc[line_gdf_local.MB_NM == xx]
                    print(mb_gdf.columns.tolist())
                else:
                    print('전국 ㄴㄴ')
                    return None
                # .loc[i,'rich_class']
                for i in mb_gdf.index.to_list():
                    ru_did = mb_gdf.loc[i,'rich_class'].ru_rch_did
                    lu_did = mb_gdf.loc[i,'rich_class'].lu_rch_did
                    

                    if ru_did is None or ru_did not in mb_gdf.RCH_DID.tolist():
                        if lu_did is None or lu_did not in mb_gdf.RCH_DID.tolist():
                            rch_list_i = []
                            rch_list_i.append(mb_gdf.loc[i,'rich_class'].PEC_est)

                            rch_len_list_i = []
                            rch_len_list_i.append(line_gdf_local.loc[line_gdf_local.RCH_DID == mb_gdf.loc[i,'rich_class'].rich_did].RCH_LEN.values[0])

                            stp_list_i = []
                            if len(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == mb_gdf.loc[i,'rich_class'].rich_did, '시설명']) == 1:
                                stp_list_i.append(stp_point_gdf_local.loc[stp_point_gdf_local.RCH_DID == mb_gdf.loc[i,'rich_class'].rich_did, '시설명'].values[0])
                            else:
                                stp_list_i.append(None)

                            ######## 이 뒤로 stp_list_i 붙이는 것 수정
                            if mb_gdf.loc[i,'rich_class'].parent is not None:
                                parent = mb_gdf.loc[i,'rich_class'].parent
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
                                title=f'PEC_ws({xx})', # Title 
                                color_discrete_map=color_discrete_map,
                                )
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

            @pn.depends(xx=select_area.param.value)
            def stream_chart(xx): 
                rich_plot=get_stream_pec_plot(xx, line_gdf, stp_point_gdf2)
                return rich_plot
                
            @pn.depends(x=radio_group3.param.value)
            def main_s(x):
                if x =='수환경 예측환경농도 입력정보':
                    tab=pn.Column(flow_1,fomu1,mark2,table_a,table4,mark2,table_b,mark2,table_c_a)
                elif x =='수생태 예측환경농도':
                    tab=pn.Column(mark5,mark8,table,pie_part,mark2,table3,mark3,mark2,page_2_3_mark,select_area,mock2,pn.Row(mock4,map_s),stream_chart,pn.Row(pec_fig,pn.Column(pec_table,pnec_ratio)))
                    # tab=pn.Column(mark5,mark8,table,pie_part,mark2,table3,mark3,mark2,page_2_3_mark,select_area,mock2,pn.Row(pec_fig,map_s),pn.Row(pec_table,pnec_ratio),mark2)
                elif x =='수환경 인체 간접노출평가 입력정보':
                    tab=pn.Column(flow_3,fomu2,mark2,figure_4,fomu3,mark2,mock3,mark2,mark9,pn.Row(weight_plot_pn,dis_water_plot_pn),mark2,dis_fish_plot_pn)
                elif x =='수환경 인체 간접 노출량 산정결과':
                    tab=pn.Column(mark10,cdf_plot_page,mark2,mark11,stack_bar,mark2,mark12,pie_page)
                return pn.Column(radio_group3,tab)

            tabs=pn.Column(main_s)
            tabs.background="#ffffff"
        elif radio_group.value=='인체노출량 산정':
            alg_name, gen_alg1, gen_alg2, gen_alg3, gen_alg4, gen_alg5, gen_alg6,p2_table_data, p3_table_data, p4_table_data,\
            p6_table_data, p13, p14, p15, p16, data, p18, alg_sum, selected_product_list,p19,p20\
            = biocide.user_input(chemi_input.value, float(text_input2.value), float(text_input5.value), 1.0, 1.0)

            input_chemical=chemi_input.value
            M= float(text_input2.value)
            P_vap = float(text_input5.value)
            der_abs = 1.0
            inh_abs = 1.0

            product_content = pd.read_csv('product_chem_final.csv', encoding = 'CP949')
            alg_list = pd.read_csv('algo_final_ver.4(final).csv', encoding = 'CP949')

            def create_Exposure_intensity() :
                text = '## * 제품별 노출강도 (경피노출, 흡입노출)'
                def make_plot(title, ax, label):
                    if len(ax) != 0:
                        p = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=600) #,x_axis_type="log"
                    else:
                        p = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=600) #,x_axis_type="log"
                    p.xaxis.axis_label = label
                    p.title.text = str(title)
                    hist, edges = np.histogram(ax, density=True, bins=30)
                    p.quad(top=hist, bottom=0, color='#E69F00', alpha=0.3, left=edges[:-1], right=edges[1:])
                    p.xaxis[0].formatter = FuncTickFormatter(code="""
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
                    p.axis.major_label_text_font = "sans"
                    return p

                column = bokeh.layouts.Column(
                    children=[_markdown(text)])
                for i in selected_product_list:
                    for j in alg_name.keys():
                        if alg_list[alg_list["code"] == i][j].values[0] == 1:
                            chart_title = alg_list[alg_list['code'] == i]['category'].values + ' ' + \
                                        alg_list[alg_list['code'] == i]['use'].values + ' ' + alg_list[alg_list['code'] == i][
                                            'type_1'].values + ' ' + alg_list[alg_list['code'] == i]['type_2'].values + ', ' + \
                                        alg_name[j] +' (노출강도)'
                            if j == 'airborne_short' :
                                label = 'mg/day/kg'
                                ax1_1 = np.log10(gen_alg1[i][gen_alg1[i] > 0])
                                p1 = make_plot(chart_title, ax1_1, label)
                                column.children.append(p1)
                            if j == "airborne_release":
                                label = '1/min/day/kg'
                                ax2_1 = np.log10(gen_alg2[i][gen_alg2[i] > 0])
                                p2 = make_plot(chart_title, ax2_1, label)
                                column.children.append(p2)
                            if j == "conti_release" :
                                label = '1/kg'
                                ax3_1 = np.log10(gen_alg3[i][gen_alg3[i] > 0])
                                p3 = make_plot(chart_title, ax3_1, label)
                                column.children.append(p3)
                            if j == "surface_volatilization" :
                                label = 'mg/day/kg'
                                ax4_1 = np.log10(gen_alg4[i][gen_alg4[i] > 0])
                                p4 = make_plot(chart_title, ax4_1, label)
                                column.children.append(p4)
                            if j == "liquid_contact":
                                label = 'mg/day/kg'
                                ax5_1 = np.log10(gen_alg5[i][gen_alg5[i] > 0])
                                p5 = make_plot(chart_title, ax5_1, label)
                                column.children.append(p5)
                            if j == "spraying_contact":
                                label = 'min/day'
                                ax6_1 = np.log10(gen_alg6[i][gen_alg6[i] > 0])
                                p6 = make_plot(chart_title, ax6_1, label)
                                column.children.append(p6)
                return column
            def Exposure_intensity():
                column = create_Exposure_intensity()
                return pn.Column(column)

            def _markdown(text):
                return bokeh.models.widgets.markups.Div(
                    text=markdown.markdown(text), sizing_mode="stretch_width"
                )

            product_content = pd.read_csv('product_chem_final.csv', encoding = 'CP949')

            def create_Distribution_exposure_product_table() :
                text = """  
                """
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p2_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_all = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p3_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_m = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=200, height=390, sizing_mode="fixed"
                )
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p4_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_w = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=200, height=390, sizing_mode="fixed"
                )

                grid = bokeh.layouts.grid(
                    children=[
                        _markdown(text),
                        [data_table_all],
                        [data_table_m],
                        [data_table_w],
                    ],
                )
                return grid


            def create_Distribution_exposure_product() :
                text = """
                """
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p2_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_all = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p3_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_m = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p4_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_w = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=p6_table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                inh_data_table_all = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                
                grid = bokeh.layouts.grid(
                    children=[
                        _markdown(text),
                        [p13,p19],
                        #[p19],
                        [_markdown(text)],
                        [p14, data_table_all],
                        [_markdown(text)],
                        [p18, inh_data_table_all],
                        [_markdown(text)],
                        [p15, data_table_m],
                        [_markdown(text)],
                        [p16, data_table_w],
                    ],
                )
                return grid

            def Distribution_exposure_product():
                grid = create_Distribution_exposure_product()
                return pn.Column(grid)


            def create_Cumulative_exposure_distribution_table(table_data) :
                text = """
                """
                data = dict(
                    col = [5,50,75,90,95,99],
                    dates=table_data,
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="분위",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                data_table_all = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
                )
                return data_table_all

            def create_Cumulative_exposure_distribution() :
                text = """
                """
                column = bokeh.layouts.column(
                    _markdown(text),
                    sizing_mode="stretch_width"
                )
                table_column = bokeh.layouts.column(
                    _markdown(text),
                    sizing_mode="stretch_width"
                )
                tot_prod = np.sum(alg_sum, axis=0)
                tot_prod_df = pd.DataFrame(tot_prod, columns=["total_exposure"])
                sort_tot_prod_df = tot_prod_df.sort_values(by=['total_exposure'], axis=0, ascending=False)
                if len(selected_product_list) < 5:
                    sort_tot_prod_df_order = sort_tot_prod_df[:len(selected_product_list)]
                    for i in range(len(selected_product_list)):
                        table_data = [format(np.percentile(
                            alg_sum[:, sort_tot_prod_df_order.index[i]][alg_sum[:, sort_tot_prod_df_order.index[i]] > 0], 5), '.2E'),
                                    format(np.percentile(alg_sum[:, sort_tot_prod_df_order.index[i]][
                                                            alg_sum[:, sort_tot_prod_df_order.index[i]] > 0], 50), '.2E'),
                                    format(np.percentile(alg_sum[:, sort_tot_prod_df_order.index[i]][
                                                            alg_sum[:, sort_tot_prod_df_order.index[i]] > 0], 75), '.2E'), format(
                                np.percentile(
                                    alg_sum[:, sort_tot_prod_df_order.index[i]][alg_sum[:, sort_tot_prod_df_order.index[i]] > 0],
                                    90), '.2E'),
                                    format(np.percentile(alg_sum[:, sort_tot_prod_df_order.index[i]][
                                                            alg_sum[:, sort_tot_prod_df_order.index[i]] > 0], 95), '.2E'), format(
                                np.percentile(
                                    alg_sum[:, sort_tot_prod_df_order.index[i]][alg_sum[:, sort_tot_prod_df_order.index[i]] > 0],
                                    99), '.2E')]
                        table = create_Cumulative_exposure_distribution_table(table_data)

                        start_date = np.log10(np.percentile(alg_sum[:, sort_tot_prod_df_order.index[i]][alg_sum[:, sort_tot_prod_df_order.index[i]] > 0], 95))
                        end_date = np.log10(1.0)

                        ax1 = np.log10(alg_sum[:, sort_tot_prod_df_order.index[i]][alg_sum[:, sort_tot_prod_df_order.index[i]] > 0])
                        ax1 = np.sort(ax1)
                        plt = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=390, x_range = (min(ax1[0], end_date) - (max(ax1[-1], end_date) - min(ax1[0], end_date)) * 0.2, max(ax1[-1], end_date) + (max(ax1[-1], end_date) - min(ax1[0], end_date)) * 0.4))
                        plt.xaxis.axis_label = '노출량 (mg/kg/day)'
                        plt.yaxis.axis_label = '빈도 (상대빈도)'
                        plt_title = product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_order.index[i]]][
                            'category'].values[0]
                        plt_title = plt_title + product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_order.index[i]]][
                            'use'].values[0]
                        plt_title = plt_title + product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_order.index[i]]][
                            'type'].values[0]
                        plt.title.text = plt_title

                        hist, edges = np.histogram(ax1, density=True, bins=50)

                        daylight_savings_start = bokeh.models.Span(location=start_date,
                                                                dimension='height', line_color='red',
                                                                line_dash='dashed', line_width=1)
                        daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                                        x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
                        plt.renderers.extend([daylight_savings_start, daylight_savings_start_label])

                        plt.quad(top=hist, bottom=0,color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])
                        plt.xaxis[0].formatter = FuncTickFormatter(code="""
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
                        plt.axis.major_label_text_font = "sans"
                        column.children.append(bokeh.layouts.grid(children=[[plt, table],[_markdown(text)]],))
                        table_column.children.append(table)
                else:
                    sort_tot_prod_df_5th = sort_tot_prod_df[:5]
                    for i in range(5):
                        table_data = [format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 5),'.2E'),
                                    format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 50),'.2E'),
                                    format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 75),'.2E'),
                                    format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 90),'.2E'),
                                    format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 95),'.2E'),
                                    format(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 99),'.2E')]
                        table = create_Cumulative_exposure_distribution_table(table_data)

                        start_date = np.log10(np.percentile(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0], 95))
                        end_date = np.log10(1.0)

                        ax1 = np.log10(alg_sum[:,sort_tot_prod_df_5th.index[i]][alg_sum[:,sort_tot_prod_df_5th.index[i]] > 0])
                        ax1 = np.sort(ax1)
                        plt = bokeh.plotting.figure(sizing_mode="fixed", width=800, height=390, x_range = (min(ax1[0], end_date) - (max(ax1[-1], end_date) - min(ax1[0], end_date)) * 0.2, max(ax1[-1], end_date) + (max(ax1[-1], end_date) - min(ax1[0], end_date)) * 0.4))
                        plt.xaxis.axis_label = '노출량 (mg/kg/day)'
                        plt.yaxis.axis_label = '빈도 (상대빈도)'
                        plt_title = product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_5th.index[i]]][
                            'category'].values[0]
                        plt_title = plt_title + product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_5th.index[i]]][
                            'use'].values[0]
                        plt_title = plt_title + product_content[product_content['code'] == selected_product_list[sort_tot_prod_df_5th.index[i]]][
                            'type'].values[0]
                        plt.title.text = plt_title

                        hist, edges = np.histogram(ax1, density=True, bins=50)

                        daylight_savings_start = bokeh.models.Span(location=start_date,
                                                                dimension='height', line_color='red',
                                                                line_dash='dashed', line_width=1)
                        daylight_savings_start_label = bokeh.models.Label(text_color=daylight_savings_start.line_color, text='95th',
                                                                        x=daylight_savings_start.location + 0.01, y=max(hist)*.15)
                        plt.renderers.extend([daylight_savings_start, daylight_savings_start_label])


                        plt.quad(top=hist, bottom=0, color='#4673eb', alpha=0.3, left=edges[:-1], right=edges[1:])

                        plt.xaxis[0].formatter = FuncTickFormatter(code="""
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
                        plt.axis.major_label_text_font = "sans"
                        column.children.append(bokeh.layouts.grid(children=[[plt, table],[_markdown(text)] ], ))
                        table_column.children.append(table)
                return column, table_column

            def Cumulative_exposure_distribution():

                column, table_column = create_Cumulative_exposure_distribution()
                return pn.Column(column)

            # def create_Contribution_product() :
            #     text = """
            #     """
            #     column = bokeh.layouts.Column(
            #         children=[_markdown(text),p17])
                return column
            def Contribution_product():
                labels = list(data.country)
                portion_list = list(data.value)
                fig = go.Figure(data=[go.Pie(labels=labels, values=portion_list, hole=.5)])
                fig.update_layout(width=720, height=500,title="제품별 전신 노출 기여도")
                return pn.Column(fig)

            def pie_inh():
                labels = list(p20.index)
                portion_list = list(p20.value)
                fig = go.Figure(data=[go.Pie(labels=labels, values=portion_list, hole=.5)])
                fig.update_layout(width=720, height=500,title="제품별 흡입 노출 기여도")
                return pn.Column(fig)
                
            def summary():
                selected_product_content = product_content[product_content["CAS"] == input_chemical]
                substance_name = selected_product_content.iloc[0, 1]

                text = '## </br></br> * 물질명 : ' + str(substance_name) + '</br></br>' + '* CAS No. : ' + input_chemical + '</br></br></br> * 물질정보'
                text1 = '## * 함유 제품 정보 (초록누리)'
                # text2 = '</br></br> * 제품별 노출량'
                # text3 = '누적 노출분포'
                # text4 = '제품별 기여도'

                data = dict(
                    col = ['분자량 [g/mole]','증기압 [Pa]','경피 흡수율 [-]','흡입 흡수율 [-]'],
                    dates=[M, P_vap, der_abs, inh_abs]
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="col", title="물질정보(단위)",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="dates", title="노출량 (mg/kg/day)",
                    ),
                ]
                Substance_information_table = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=400, height=390, sizing_mode="fixed"
                )

                data = dict(
                    product_name = selected_product_content["product_name"],
                    category = selected_product_content["category"],
                    type = selected_product_content["type"],
                    conc_min = selected_product_content["conc_min"],
                    conc_max = selected_product_content["conc_max"],
                )
                source = bokeh.models.ColumnDataSource(data)

                columns = [
                    bokeh.models.widgets.TableColumn(
                        field="product_name", title="product_name",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="category", title="category",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="type", title="type",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="conc_min", title="함유량(최소) (%)",
                    ),
                    bokeh.models.widgets.TableColumn(
                        field="conc_max", title="함유량(최대) (%)",
                    ),
                ]

                product_information_table = bokeh.models.widgets.DataTable(
                    source=source, columns=columns, width=700, height=390, sizing_mode="fixed"
                )

                text_column1 = _markdown(text)
                text_column2 = _markdown(text1)
                # text_column3 = _markdown(text2)
                # text_column4 = _markdown(text3)
                # text_column5 = _markdown(text4)
                #column1 = create_Exposure_intensity()
                # column2 = create_Distribution_exposure_product_table()
                # column3, table_column = create_Cumulative_exposure_distribution()
                # column4 = create_Contribution_product()

                layout = bokeh.layouts.layout([
                    [text_column1],
                    [Substance_information_table],
                    [text_column2],
                    [product_information_table],
                    # [text_column3],
                    # [column2],
                    # [text_column4],
                    # [table_column],
                    # [text_column5],
                    # [column4],
                ])

                return pn.Column(layout)
            mark13=pn.pane.Markdown("<brt><br>")
            flow_4=pn.Column(pn.pane.JPG('소지바제품 누적 통합 노출량 산정 절차 그림.jpg',height=470,width=800),pn.pane.JPG('노출강도 설명 그림.jpg',height=470,width=800))
            # tabs=pn.Tabs(
            #     ('입력정보확인', pn.Column(flow_4,summary(),mark13,Exposure_intensity())),
            #     ('누적노출분포', Distribution_exposure_product()),
            #     ('제품별노출분포',Cumulative_exposure_distribution()),
            #     ('제품별기여도',pn.Column(Contribution_product(),pie_inh())),
            #     dynamic=True
            # )


            @pn.depends(x=radio_group4.param.value)
            def main_s(x):
                if x =='입력정보확인':
                    tab=pn.Column(flow_4,summary(),mark13,Exposure_intensity())
                elif x=='누적노출분포':
                    tab=Distribution_exposure_product()
                elif x=='제품별노출분포':
                    tab=Cumulative_exposure_distribution()
                elif x=='제품별기여도':
                    tab=pn.Column(Contribution_product(),pie_inh())
                return pn.Column(radio_group4,tab)
            tabs=pn.Column(main_s)
            tabs.background="#ffffff"
    return tabs
mark_input=pn.pane.Markdown("### * 계산 버튼을 누르기전에 <br> 파일을 업로드 했는지 확인해주세요", style={'font-family': 'NanumBarunGothic','font-size':'25px'})

# template = pn.template.FastListTemplate(
#     site="EHR&C", title="살생물제 및 생활화학제품 사용에 따른 생태 및 인체노출량 산정 프로그램" ,
    
#     sidebar=[selcet_input,mark,search_chemi,mark_input,mark2,select_cami],
#     main=[calculate_A_batch],
#     font="NanumBarunGothic",
#     header_background='#2edd2b',
#     background_color="#ffffff",
#     theme_toggle=False,
#     # accent_base_color="#fffcfc",
#     neutral_color="#ffffff"
#     # theme="dark"
#     # shadow=False
#     # main=[area]
# )
template = pn.template.MaterialTemplate(
    site="EHR&C", title="살생물제 및 생활화학제품 사용에 따른 생태 및 인체노출량 산정 프로그램" ,    
    sidebar=[selcet_input,mark,search_chemi,mark_input,mark2,select_cami],
    main=[calculate_A_batch],
    header_background='#2edd2b',
    theme_toggle=False,
    # accent_base_color="#fffcfc",
)

template.sidebar_width=1000
template.main_max_width='150%'
template.servable()
