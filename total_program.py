from pathlib import Path
import pandas as pd
import panel as pn
# import dask.dataframe as dd
import numpy as np
from io import BytesIO
import random
import plotly.express as px
import plotly.io as pio
import random
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib
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
path = os.getcwd()
os.getcwd()

css = [
        'https://cdn.datatables.net/1.10.24/css/jquery.dataTables.min.css',
            # Below: Needed for export buttons
        'https://cdn.datatables.net/buttons/1.7.0/css/buttons.dataTables.min.css',
        
       ]

js = {
    '$': 'https://code.jquery.com/jquery-3.5.1.js',
    'DataTable': 'https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js',
    # Below: Needed for export buttons
    'buttons': 'https://cdn.datatables.net/buttons/1.7.0/js/dataTables.buttons.min.js',
    'jszip': 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js',
    'pdfmake': 'https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js',
    'vfsfonts': 'https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js',
    'html5buttons': 'https://cdn.datatables.net/buttons/1.7.0/js/buttons.html5.min.js',
}
pio.renderers.default='notebook'
pn.extension('plotly','tabulator','katex', 'mathjax',sizing_mode='stretch_width',css_files=css, js_files=js)
pd.options.display.float_format = '{:.1E}'.format

list_1=pd.read_csv("604chemical_default.csv", thousands = ',')
cas_rn=list_1['CAS_Num']+" "+list_1['CHEM']
cas_rn_val=cas_rn.values
cas_rn_val=cas_rn_val.tolist()
cas_rn_val.insert(0,'')
optionss=cas_rn_val.copy()

select_cami=pn.widgets.Select(name="화학물질_리스트", options=optionss, value='', sizing_mode='fixed')

chemi_input= pn.widgets.TextInput(name='화학물질 입력', sizing_mode='fixed',width=150) 
button3 = pn.widgets.Button(name='검색', button_type='primary',sizing_mode='fixed',width=150)

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
text_input = pn.widgets.TextInput(name='물질명', sizing_mode='fixed',width=350)
text_input2 = pn.widgets.TextInput(name='분자량 (g/mol)',sizing_mode='fixed',width=120)
text_input3 = pn.widgets.TextInput(name='녹는점(℃)', sizing_mode='fixed',width=120)
text_input4 = pn.widgets.TextInput(name='옥탄올-물 분배계수', sizing_mode='fixed',width=120)
text_input5 = pn.widgets.TextInput(name='증기압 (at 25℃) (Pa)', sizing_mode='fixed',width=120)
text_input6 = pn.widgets.TextInput(name='물용해도 (at 25℃) (㎎/L)', sizing_mode='fixed',width=120)
select_model=pn.widgets.Select(name='이분해성', options=options2, value='', sizing_mode='fixed')
text_input8 = pn.widgets.TextInput(name='Koc (L/㎏)', sizing_mode='fixed',width=120,margin=(0,0,20,10))
# text_input10 = pn.widgets.TextInput(name='PNEC', sizing_mode='fixed',width=120,margin=(0,0,20,10))
widget_box2=pn.Column(text_input, text_input2, text_input3, text_input4, text_input5,text_input6, select_model, text_input8)

radio_group = pn.widgets.RadioBoxGroup(name='RadioBoxGroup', options=['수생태 환경노출량 산정', '인체노출량 산정'], inline=False)
file_input = pn.widgets.FileInput(accept='.csv,.json',name='유통량 자료 업로드',sizing_mode='fixed')
# file_input2 = pn.widgets.FileInput(accept='.csv,.json',name='사용량 자료 업로드',sizing_mode='fixed')

radio_group3 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['수환경 예측환경농도 입력정보', '수생태 예측환경농도', '수환경 인체 간접노출평가 입력정보','수환경 인체 간접 노출량 산정결과'], sizing_mode='stretch_width', button_type='success',margin=(0,0,50,0))

radio_group4 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group2', options=['입력정보확인', '누적노출분포', '제품별노출분포','제품별기여도'], button_type='success',margin=(0,0,50,0))

button = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed')
button2 = pn.widgets.Button(name='Refresh', button_type='primary',sizing_mode='fixed',width=120)

mark3=pn.pane.Markdown("### [제품 정보 입력]", sizing_mode='fixed', margin=(0, 0, 20, 0), style={'font-family': "NanumBarunGothic"})
mark4=pn.pane.Markdown("### [물질량 정보 입력]", sizing_mode='fixed', margin=(0, 0, 20, 0), style={'font-family': "NanumBarunGothic"})
mark5=pn.pane.Markdown("<br>")
file_download = pn.widgets.FileDownload(file='입력자료 템플릿.csv', filename='입력자료 템플릿.csv',sizing_mode='fixed',width=250,margin=(25,0,25,0))
# file_download2 = pn.widgets.FileDownload(file='사용량 자료 템플릿.csv', filename='사용량 자료 템플릿.csv',sizing_mode='fixed',width=250,margin=(25,0,25,0))
file_input = pn.widgets.FileInput(accept='.csv,.json',name='유통량 자료 업로드',sizing_mode='fixed')

value_option1='3.068'
value_option2='54.520'

radio_group2 = pn.widgets.RadioBoxGroup(name='RadioBoxGroup', options=['입력자료 갯수 5개 이하', '입력자료 업로드'], inline=False)
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
button4=pn.widgets.Button(name='입력', button_type='primary',sizing_mode='fixed',width=120,margin=(23,0,0,10))
input_self=pn.Row(text_input21,button4)
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
        widget_box3=pn.Column(input_self,mark5,input_kind)
    elif x =='입력자료 업로드':
        widget_box3=pn.Column(file_download,file_input)
    return widget_box3
@pn.depends(x=radio_group.param.value)
def radio_option(x):
    marks=pn.pane.Markdown("## <br> ■  입력 자료 갯수 선택 <br>", style={'font-family': "NanumBarunGothic"})
    if x == '수생태 환경노출량 산정':
        widget_box3=pn.Column(marks,radio_group2,mark5,radio_option2,mark5)
    elif x =='인체노출량 산정':
        widget_box3=pn.Column()
    return widget_box3
mark=pn.pane.Markdown('<br>')
mark2=pn.pane.Markdown("#### 본 프로그램에서 사용가능한 CAS넘버는 아래의 리스트에서 확인 가능합니다.", style={'font-family': "NanumBarunGothic"})

widget_box=pn.Column(pn.pane.Markdown('<br>'),pn.pane.Markdown('## ■ 물성정보확인 (물성정보 수정시, 직접입력)', style={'font-family': "NanumBarunGothic"}),widget_box2,pn.pane.Markdown('## ■ 생태 및 인체 노출량 산정방식 선택', style={'font-family': "NanumBarunGothic"}),radio_group,radio_option,button)

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

mark7=pn.pane.Markdown("#### - 생산제품중 물질량 : 연간 국내에서 생산된 제품에 포함된 물질의 총량 <br>  - 수입제품중 물질량 : 국외에서 생산되어 연간 국내로 수입된 제품에 포함된 물질의 총량 <br>   - 국내유통 물질량 : 국내생산제품에 포함된 총 물질량과 국내수입제품에 포함된 총 물질량의 합 (제품 사용단계에서 모두 환경으로 배출된다고 가정) ", style={'font-family': "NanumBarunGothic"})

mark8=pn.pane.Markdown("#### ■ 물질 제조단계 및 제품 생산단계 환경배출량 <br>   - 물질의 제조단계 및 제품의 생산단계에서 배출되는 물질의 양 (제조 및 생산 공정별 배출계수 반영) <br>  -  공정에 대한 정보가 없는 경우 보수적으로 접근 : 물질 제조단계에서 배출계수 ERC1 (대기 5%, 수계 6%, 토양 0.01%), 제품 생산단계에서 배출계수 ERC2 (2.5%, 2%, 0.01%) <br><br> ■ 제품 사용단계 환경배출량 <br>  -  제품에 포함된 물질량에 제품별 배출계수를 곱해 제품 사용시 환경으로 배출되는 물질량을 매체별로 계산하고, 전체 제품군에 대해 매체별 배출량을 합산한 값 <br><br> ■ 1인 1일 평균 물질사용량 : 연간 국내에서 유통되는 제품에 포함된 총 물질량을 국내 전체 인구수와 연간 사용일수(365일)로 나눈 값", style={'font-family': "NanumBarunGothic"})
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
        # tts=pn.pane.Markdown("## ■ 본 프로그램은 아래의 목적으로 제작되었습니다. <br> 1.  생활화학제품(소비자제품)에 포함된 유해물질이 제품생산단계와 제품사용단계에서 환경으로 배출되어 발생하는 수계 노출량 산정 <br> 2.  이후 환경 중 거동으로 음용수와 담수어패류에 전이되어 발생하는 간접 인체노출량 산정 <br> 3.  생활화학제품(소비자제품) 사용시 발생하는 직접 인체노출량 산정 <br><br> ■ 본 결과물은 환경부의 재원으로 한국환경산업기술원 생활화학제품 안전관리 기술개발사업『제품 함유 유해물질 수생태 환경 노출지수 개발』의 지원을 받아 연구되었습니다. (2020002970009, 1485017560) <br> <br> ■ This work was supported by Korea Environment Industry & Technology Institute(KEITI) through Technology Development Project for Safety Management of Household Chemical Products 『Development of aquatic environment exposure index for hazardous substances containing products』, funded by Korea Ministry of Environment(MOE) (2020002970009, 1485017560)")
        tabs=pn.Column(pn.pane.JPG('수생태 노출량 평가 모델 첫 화면_new.jpg',height=560,width=1100,margin=(0,0,50,0)))
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
            df3.columns=['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식','생산제품중_물질량 (톤/년)','수입제품중_물질량 (톤/년)','총 국내유통_ 물질량 (톤/년)','배출계수_대기','배출계수_수계','배출계수_토양','배출계수_대기_사용','배출계수_수계_사용','배출계수_토양_사용']
            df3_t=df3[['배출계수_대기','배출계수_수계','배출계수_토양','배출계수_대기_사용','배출계수_수계_사용','배출계수_토양_사용']]
            df3_t.columns=['물질제조 · 제품생산단계 배출계수_대기','물질제조 · 제품생산단계 배출계수_수계','물질제조 · 제품생산단계 배출계수_토양','제품사용단계 배출계수_대기','제품사용단계 배출계수_수계','제품사용단계 배출계수_토양']


            flow_1=pn.Column(pn.pane.Markdown("## Ⅰ-1. 하수종말처리장 방류지점의 예측환경농도(PEC<sub>local_water</sub>) 산정방법", style={'font-family': "NanumBarunGothic"}),pn.pane.JPG('1page_new.jpg',height=470,width=600,margin=(0,0,50,0)))

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

            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable({
                lengthChange:false,
            });
            } else {
            $(document).ready(function () {
                $('.example').DataTable({
                lengthChange:false,
            });
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            script2 = """
            <script>
            if (document.readyState === "complete") {
            $('.example2').DataTable({
                lengthChange:false,
            });
            } else {
            $(document).ready(function () {
                $('.example2').DataTable({
                lengthChange:false,
            });
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            
            script3 = """
            <script>
            if (document.readyState === "complete") {
            $('.example3').DataTable({
                lengthChange:false,
            });
            } else {
            $(document).ready(function () {
                $('.example3').DataTable({
                lengthChange:false,
            });
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """

            script4 = """
            <script>
            if (document.readyState === "complete") {
            $('.example4').DataTable({
                lengthChange:false,
            });
            } else {
            $(document).ready(function () {
                $('.example4').DataTable({
                lengthChange:false,
            });
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """

            # df_data=df05_table1.loc[[chemical]]
            df_data=df2.loc[[chemical]]
            df_data_2=df_data.copy()
            df_data_2=df_data_2.reset_index(drop=True)
            df_data_2=df_data_2.set_index('분류')
            html2 = df_data_2.to_html(classes=['example2', 'panel-df'])
            table_a=pn.Column(pn.pane.Markdown("## Ⅰ-2. 물질함유 제품정보 ("+str(chemical)+")", style={'font-family': "NanumBarunGothic"}),mark7,pn.pane.HTML(html2+script2,width=1200,height=200,margin=(0,0,200,0)))

            df3_t=df3_t.loc[[chemical]]
            df3_s=df_data.copy()
            df3_s=df3_s[['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식']]
            df3_s_t=pd.concat([df3_s,df3_t],axis=1)
            df3_s_t=df3_s_t.reset_index(drop=True)
            df3_s_t=df3_s_t.set_index('분류')
            html3 = df3_s_t.to_html(classes=['example3', 'panel-df'])
            table_b=pn.Column(pn.pane.Markdown("## Ⅰ-3. 물질제조 · 제품생산 및 제품사용단계 배출계수 ("+str(chemical)+")", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html3+script3,width=1500,height=200,margin=(0,0,200,0)))

            stp_removal_df=pd.DataFrame({'CAS-RN':[str(chemical)],'STP 제거율(%)':[float(STP_removal_pecent)]})
            stp_removal_df=stp_removal_df.set_index('CAS-RN')
            # df01_table=df01[['STP_removal']]
            # df01_table=df01_table.loc[[chemical]]
            # df01_table=df01_table.rename(columns={'STP_removal':'STP 제거율(%)'})
            table_c=pn.Column(pn.pane.Markdown("## Ⅰ-4. 물질의 STP 제거율 정보", style={'font-family': "NanumBarunGothic"}),stp_removal_df,width=300,height=200)
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

            table=pn.Column(pn.pane.Markdown("### * 단계별 환경배출량", style={'font-family': "NanumBarunGothic"}),dd2,width=600,height=200,margin=(0,0,80,0))
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
            domes_table=pn.Column(domestic_background_conc_data,pn.pane.Markdown("<br>"),width=500,height=100)
            mark3=pn.pane.Markdown(" #### ■ 매체별 전국 배경농도 <br>   -  제조 · 생산단계에서 환경으로 배출된 물질량과 <br> 사용단계에서 환경 매체별로 배출된 물질량의 합을 입력자료로 하여 <br> SimpleBox Korea 기반으로 예측 <br><br> ■ 수계 지점별 예측환경농도 <br>   - Tier 1 : 특정 지점에서 배출되는 물질량으로 STP 방류구역의 화학물질 농도를 예측하고, 전국 배경농도와 합산하여 수계 특정 지점의 예측환경농도를 산정 <br><br>   - Tier 2 : 특정 지점에서 배출되는 물질량으로 STP 방류구역의 화학물질 농도를 예측하고, 상류에서 배출된 물질의 누적을 고려한 배경농도와 합산하여 수계 특정 지점의 예측환경농도를 산정", style={'font-family': "NanumBarunGothic"})
            table3=pn.Column(pn.pane.Markdown("## Ⅱ-2. 매체별 전국 배경농도", style={'font-family': "NanumBarunGothic"}),domes_table)

            e_data=e.to_frame(name='1인 1일 물질사용량 (g/day · 명)')
            e_data=e_data.loc[[chemical]]
            df_e=df_data.copy()
            df_e=df_e[['분류','살생물제품유형','제형','사용장소','적용방식','후처리방식']]
            df_e_t=pd.concat([df_e,e_data],axis=1)
            df_e_t=df_e_t.reset_index(drop=True)
            df_e_t=df_e_t.set_index('분류')
            html4 = df_e_t.to_html(classes=['example4', 'panel-df'])
            table4=pn.Column(pn.pane.Markdown("### * 1인 1일 물질사용량 (g/day · 명)", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html4+script4,width=1000,height=200,margin=(0,0,200,0)))
            # table4=pn.Column(pn.pane.Markdown("### * 1인 1일 물질사용량 (g/day · 명)"),e_data,width=500,height=200,margin=(0,0,50,0))
#############################################

            # index 같은것만 고름
            stp_removal = STP_removal_pecent
##########################################################

############################################################
            def take_map_df():
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

                if chemical == '007173-51-5':
                    k = 0.00193 # 반응속도상수 반감기 역수 h-1 #추후 변경
                elif chemical == '001222-05-5':
                    k= 0.000481 # 반응속도상수 반감기 역수 h-1 #추후 변경
                elif chemical == '002634-33-5': 
                    k = 0.00193 # 반응속도상수 반감기 역수 h-1 #추후 변경
                else:
                    k = 0.00193 # 반응속도상수 반감기 역수 h-1 #추후 변경                

                koc = float(text_input8.value) # 사용자 입력값, L/kg #KOC 있고 text_input8
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


                # 희석배율 1보다 작은지점에 방류량 추가요청
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
                return line_gdf_last, test, total_df, bpr_pec


            # 그래프그릴 데이터 t_df
            line_gdf, stp_point_gdf, t_df, b_pec = take_map_df()
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
            options2.insert(0,'전체')
            select_area=pn.widgets.Select(name='하천 지역선택', options=options2, value='전체', sizing_mode='fixed',margin=(0,1450,20,0))

            page_2_3_mark=pn.pane.Markdown(" ## Ⅱ-3. 하수처리장별 예측환경농도", style={'font-family': "NanumBarunGothic"})
            @pn.depends(xx=select_area.param.value)
            def mock2(xx):
                if xx =='전체':
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example').DataTable({
                        lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example').DataTable({
                        lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = t_df_2.to_html(classes=['example', 'panel-df'])
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ STP 방류수역 예측환경농도", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html+script,sizing_mode='fixed',width=1200,height=300,margin=(0,0,95,0)))
                else: 
                    t_df_2_t_data=t_df_2.loc[[xx]]
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example').DataTable({
                    lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example').DataTable({
                    lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = t_df_2_t_data.to_html(classes=['example', 'panel-df'])
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ STP 방류수역 예측환경농도", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html+script,sizing_mode='fixed',width=1200,height=300,margin=(0,0,95,0)))
                return table_n

            t_dfs.columns =['중권역명','RCH_DID','PEC_WS(㎎/L)',]
            t_dfs=t_dfs.set_index('중권역명')
            @pn.depends(xx=select_area.param.value)
            def mock4(xx):
                if xx =='전체':
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example2').DataTable({
                        lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example2').DataTable({
                        lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = t_dfs.to_html(classes=['example2', 'panel-df'])
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ 표준유역 예측환경농도", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html+script,sizing_mode='fixed',width=300,height=350,margin=(0,0,95,0)))
                else: 
                    t_dfs_t=t_dfs.loc[[xx]]
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example2').DataTable({
                    lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example2').DataTable({
                    lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = t_dfs_t.to_html(classes=['example2', 'panel-df'])
                    table_n=pn.Column(pn.pane.Markdown(" ### ■ 표준유역 예측환경농도", style={'font-family': "NanumBarunGothic"}),pn.pane.HTML(html+script,sizing_mode='fixed',width=300,height=350,margin=(0,0,95,0)))
                return table_n


            @pn.depends(xx=select_area.param.value)
            def pec_df(xx):
                t_df_sort=t_df.copy()
                t_df_sort.columns =['중권역명','STP명','방류량','인구수','구간체류시간(h)','희석배율','C_local','PEC_local','PEC_WS']
                t_df_sort=t_df_sort.set_index('중권역명')
                t_df_sort=t_df_sort.loc[[xx]]
                t_df_sort=t_df_sort.reset_index()
                return t_df_sort

            @pn.depends(xx=select_area.param.value)
            def pec_fig(xx):
                if xx =='전체':
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
                fig.update_layout(title=chemical_i_want+"의 수계 지점별 예측환경농도(PEC) 분포",title_font_family="NanumBarunGothic")
                fig.add_vline(x=b_pec, line_width=2, line_dash="dash", line_color="red")
                if text_input9.value=='':
                    None
                else:
                    fig.add_vline(x=float(text_input9.value), line_width=2, line_dash="dash", line_color="green")
                fig.add_hrect(y0=0.9, y1=0.95, line_width=2, line_color="red",fillcolor="red", opacity=0.2)
                return pn.Column(fig, width=650, height=500,margin=(0,50,20,0))           


            text_input9 = pn.widgets.TextInput(name='예측무영향농도(PNEC) (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed',width=120)
            button_t = pn.widgets.Button(name='입력', button_type='primary',sizing_mode='fixed',width=120)
            output1 = pn.widgets.TextInput(name='표준유역 중 PNEC 초과비율(%)',value='',disabled=True,sizing_mode='fixed',width=120)
            output2 = pn.widgets.TextInput(name='STP 방류수역 중 PNEC 초과비율(%)',value='',disabled=True,sizing_mode='fixed',width=120)

            #표준유역
            def calculate_ratio():
                if select_area.value =='전체':
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
                if select_area.value =='전체':
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

            pnec_ratio=pn.Column(pn.Row(text_input9,button_t),output1,output2)

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
                return pn.Column(table_pec,width=300,height=200)
            #%% 민승이가 말한거, fish  BAF 구하기
            N_FCODE3 = np.array([11479,11193,11707,11524,11162,11252,11289,11255,11423,11348,11690,11420]).astype('str') #물고기 코드
            N_FCODE3 = np.sort(N_FCODE3)
            weight = np.array([0.003,0.01,0.003,0.003,1,1,1,0.01,5,3,0.5,0.01]) # 물고기 weight
            Kow_list = [np.log10(float(text_input4.value))] # 물질 특성 값 5836-29-3    , 7173-51-5    , 72963-72-5

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

            options3=list(set(material_conc.index))
            options3.insert(0,'전체')
            select_area2=pn.widgets.Select(name='하천 지역선택', options=options2, value='전체', sizing_mode='fixed',margin=(0,1450,20,0))

            @pn.depends(xx=select_area2.param.value)
            def mock3(xx):
                if xx =='전체':
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example').DataTable({
                        lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example').DataTable({
                        lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = material_conc.to_html(classes=['example', 'panel-df'])
                    table_n=pn.Column(select_area2,pn.pane.HTML(html+script,width=1200,height=300,margin=(0,0,95,0)))
                else: 
                    material_conc_data=material_conc.loc[[xx]]
                    script = """
                    <script>
                    if (document.readyState === "complete") {
                    $('.example').DataTable({
                    lengthChange:false,
                    });
                    } else {
                    $(document).ready(function () {
                        $('.example').DataTable({
                    lengthChange:false,
                    });
                    })
                    }
                    document.oncontextmenu=function(){return false;}
                    document.onselectstart=function(){return false;}
                    document.ondragstart=function(){return false;}
                    </script>
                    """
                    html = material_conc_data.to_html(classes=['example', 'panel-df'])
                    table_n=pn.Column(select_area2,pn.pane.HTML(html+script,width=1200,height=300,margin=(0,0,95,0)))
                return table_n

            Con_2 = con_i_want_T.iloc[:,step2_idx].mean().mean()
            Con_3 = con_i_want_T.iloc[:,step3_idx].mean().mean()
            Con_4 = con_i_want_T.iloc[:,step4_idx].mean().mean()
            Con_w = drinking_water_T.mean().item()

            Con_2=format(Con_2,'.1E')
            Con_3=format(Con_3,'.1E')
            Con_4=format(Con_4,'.1E')
            Con_w=format(Con_w,'.1E')

            # table_n=pd.DataFrame({'구분':['물질농도'],'음용수(μg/L)':[Con_w],'영양단계2(μg/g)':[Con_2],'영양단계3(μg/g)':[Con_3],'영양단계4(μg/g)':[Con_4]})
            # # table_n=table_n.set_index('물질명')
            # table_n=table_n.style.hide_index()
            # # mark4=pn.pane.Markdown("#### ■ 음용수 및 어패류 중 물질농도 <br>   -  특정 지점의 수계 예측환경농도와 영양단계별 생물축적계수를 이용하여 음용수와 어패류에 포함된 화학물질의 농도를 예측하고, <br> 인구집단의 음용수/어패류 섭취량과 체중정보를 이용하여 수계로부터 유래되는 간접 인체노출량을 산정")
            figure_4=pn.Column(pn.pane.Markdown("## Ⅲ-2. 음용수 및 어패류 중 물질농도", style={'font-family': "NanumBarunGothic"}))
            # table_w=pn.Column(table_n,width=500,height=200,margin=(0, 50, 20, 0))
            # table_w_a=pn.Column(pn.pane.Markdown("## Ⅲ-2. 음용수 및 어패류 중 물질농도"),figure_4,table_w)

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

            age1='1세'
            age2='2~3세'
            age3='4~6세'
            age4='7~12세'
            age5='청소년 및 성인'
            age6='노출기여도'

            # # 체중
            def weight_plot(x,y):
                x1=list(x)
                hist_data = [x1]
                group_labels = [y]
                colors = ['#333F44']

                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
                fig.update_xaxes(title="체중 (㎏)",title_font_family="NanumBarunGothic")
                fig.update_yaxes(title="빈도 (-)",title_font_family="NanumBarunGothic")
                fig.update_layout(title_text=y,title_font_family="NanumBarunGothic",width=350,height=350,showlegend=False)
                return fig

            def weight_plot_pn():
                fig1= weight_plot(intake_db.iloc[intake_db_A_idx,:]['HE_wt'],age1)
                fig2= weight_plot(intake_db.iloc[intake_db_B_idx,:]['HE_wt'],age2)
                fig3= weight_plot(intake_db.iloc[intake_db_C_idx,:]['HE_wt'],age3)
                fig4= weight_plot(intake_db.iloc[intake_db_D_idx,:]['HE_wt'],age4)
                fig5= weight_plot(intake_db.iloc[intake_db_E_idx,:]['HE_wt'],age5)  
                return pn.Column(pn.pane.Markdown("### 체중", style={'font-family': "NanumBarunGothic"}),pn.Row(fig1,fig2,fig3,fig4,fig5))

            # # 음용수
            def dis_water_plot(x,y):
                x1=list(x)
                hist_data = [x1]
                group_labels = [y]
                colors = ['#333F44']

                fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
                fig.update_xaxes(title="음용수 섭취량 (L/day)",title_font_family="NanumBarunGothic")
                fig.update_yaxes(title="빈도 (-)",title_font_family="NanumBarunGothic")
                fig.update_layout(title_text=y,title_font_family="NanumBarunGothic",width=350,height=350,showlegend=False)
                return fig

            def dis_water_plot_pn():
                fig1= dis_water_plot(intake_db.iloc[intake_db_A_idx,:]['intake.w_liter.per.day'],age1)
                fig2= dis_water_plot(intake_db.iloc[intake_db_B_idx,:]['intake.w_liter.per.day'],age2)
                fig3= dis_water_plot(intake_db.iloc[intake_db_C_idx,:]['intake.w_liter.per.day'],age3)
                fig4= dis_water_plot(intake_db.iloc[intake_db_D_idx,:]['intake.w_liter.per.day'],age4)
                fig5= dis_water_plot(intake_db.iloc[intake_db_E_idx,:]['intake.w_liter.per.day'],age5)  
                return pn.Column(pn.pane.Markdown("### 음용수 섭취량", style={'font-family': "NanumBarunGothic"}),pn.Row(fig1,fig2,fig3,fig4,fig5),mark2)

            def dist_fish_list(x):
                x=pd.Series(x)
                x=list(x)
                return x

            def dis_fish_plot(x,x1,x2,y):
                test=pd.DataFrame({"영양단계2":x,"영양단계3":x1,"영양단계4":x2})
                fig=test.hvplot.kde(title=y,xlabel="어패류 섭취량 (g/day)",ylabel="빈도 (-)",xlim=(0,5)).opts(legend_position='top_right')
                return fig

            def dis_fish_plot_pn():
                fig1= dis_fish_plot(dist_fish_list(T_if2[intake_db_A_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_A_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_A_idx,:].sum(axis=1)),age1)
                fig2= dis_fish_plot(dist_fish_list(T_if2[intake_db_B_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_B_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_B_idx,:].sum(axis=1)),age2)
                fig3= dis_fish_plot(dist_fish_list(T_if2[intake_db_C_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_C_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_C_idx,:].sum(axis=1)),age3)
                fig4= dis_fish_plot(dist_fish_list(T_if2[intake_db_D_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_D_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_D_idx,:].sum(axis=1)),age4)
                fig5= dis_fish_plot(dist_fish_list(T_if2[intake_db_E_idx,:].sum(axis=1)),dist_fish_list(T_if3[intake_db_E_idx,:].sum(axis=1)),dist_fish_list(T_if4[intake_db_E_idx,:].sum(axis=1)),age5)
                return pn.Column(pn.pane.Markdown("### 어패류 섭취량", style={'font-family': "NanumBarunGothic"}),pn.Row(fig1,fig2,fig3,fig4,fig5),mark2)       


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
            # # In[] 누적분포함수

            mark9=pn.pane.Markdown("## Ⅲ-3. 연령별 인구집단의 노출계수 분포", style={'font-family': "NanumBarunGothic"})
            mark10=pn.pane.Markdown("## Ⅳ-1. 연령별 수계 유래 간접 인체노출량 plot", style={'font-family': "NanumBarunGothic"})
            mark11=pn.pane.Markdown("## Ⅳ-2. 연령별 수계 유래 간접 인체노출량 누적 bar plot", style={'font-family': "NanumBarunGothic"})
            mark12=pn.pane.Markdown("## Ⅳ-3. 연령별 음용수/어패류 섭취 노출기여도 pie chart", style={'font-family': "NanumBarunGothic"})

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
                mark_f=pn.pane.Markdown("### 어패류 섭취 노출량", style={'font-family': "NanumBarunGothic"})
                mark_w=pn.pane.Markdown("### 음용수 섭취 노출량", style={'font-family': "NanumBarunGothic"})
                mark_a=pn.pane.Markdown("### 전체(어패류 + 음용수)", style={'font-family': "NanumBarunGothic"})
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

            mark5=pn.pane.Markdown("## Ⅱ-1. 단계별 환경배출량", style={'font-family': "NanumBarunGothic"})
            flow_3=pn.Column(pn.pane.Markdown("## Ⅲ-1 수환경 인체 간접노출량 산정방법", style={'font-family': "NanumBarunGothic"}),pn.pane.JPG('FF_exp.jpg',height=470,width=800,margin=(0,0,50,0)))

            def map_s():
                ## 지도그리는 함수부분 ##
                color_list_1 = stp_point_gdf.PEC_local.values.tolist() + line_gdf.PEC_WS.tolist()
                color_list_1.sort()

                # index_list = [0.000002,0.000004,0.0000045,0.00000468,0.00000481,0.00000513,0.00000569,0.00000704,0.00000809,0.0000111,0.0000196,]
                index_list = [0.000002,0.000004,0.0000045,0.00000468,0.00000481,]
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


                m = line_gdf.explore(tiles = "CartoDB positron", column = 'PEC_WS', legend = True, cmap = con_step,)
                stp_point_gdf.explore(m=m, column = 'PEC_local', cmap = con_step, marker_type = folium.Circle(radius=1000, fill = 'white'),)
                pp=pn.panel(m, width=900,height=450,margin=(80,50,20,50))
                return pp

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
                fig.update_layout(width=400,height=400)
                fig.update_layout(title='전체 환경배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig2 = go.Figure(data=[go.Pie(labels=labels, values=portion_list2, hole=.5)])
                fig2.update_traces(marker=dict(colors=colors))
                fig2.update_layout(width=400,height=400)
                fig2.update_layout(title='대기배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig3 = go.Figure(data=[go.Pie(labels=labels, values=portion_list3, hole=.5)])
                fig3.update_traces(marker=dict(colors=colors))
                fig3.update_layout(width=400,height=400)
                fig3.update_layout(title='수계배출량 (톤/년)',title_font_family="NanumBarunGothic")
                fig4 = go.Figure(data=[go.Pie(labels=labels, values=portion_list4, hole=.5)])
                fig4.update_traces(marker=dict(colors=colors))
                fig4.update_layout(width=400,height=400)
                fig4.update_layout(title='토양배출량 (톤/년)',title_font_family="NanumBarunGothic")
                # return pn.Column(pn.Row(fig,fig2),pn.Row(fig3,fig4))
                return pn.Row(fig,fig2,fig3,fig4,sizing_mode="fixed",margin=(10,10,10,10))

            ###### 수식 마크다운들
###### 1 페이지
            title_1 = pn.pane.Markdown("""
            -------------------                           
            * **STP 방류수 농도 산정식**
            <br>
            """,sizing_mode="fixed", margin=(15,0,20,15),width=650)

            a = pn.pane.LaTeX(r"""
            $\begin{aligned} & {C_{local}}_{eff} = \frac {{E_{local}}_{water} \times 10^6 \times (1-STP_{removal} \times 100)}{EFFLUENT_{stp}} & \end{aligned} $
            """, style={'font-size': '20pt'},sizing_mode="fixed", margin=(15,0,20,15),width=650)

            defi_1 = pn.pane.Markdown("""
            용어: 
            <blockquote>
            <p> $${C_{local}}_{eff}$$ (㎎/ℓ) : STP 방류수 중 물질농도</p>
            <p> $${E_{local}}_{water}$$ (㎏/day) : 하수로 배출되는 물질량 (=원단위 x 인구수)</p>
            <p> $$STP_{removal}$$ (%) : STP 처리과정에서의 물질제거율</p>
            <p> $$EFFLUENT_{stp}$$ (ℓ/day) : STP 일일 방류량</p>
            </blockquote>
            """,sizing_mode="fixed", margin=(15,0,20,15),width=650)

            page_1=pn.Column(title_1,a,defi_1,margin=(0,0,0,0))

            title_2 = pn.pane.Markdown("""
                -------------------                           
                * **STP 방류지점의 예측환경농도 산정식**
                """,sizing_mode="fixed", margin=(15,0,20,15),width=650)

            b = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & {PEC_{local}}_{water} = \frac {{C_{local}}_{eff}}{(1+{F_{oc}}_{susp} \cdot K_{oc}\cdot SUSP_{water} \cdot 10^{-6} \cdot DILUTION)} + PEC_{regional} & \end{aligned}$
            """, style={'font-size': '15pt'},sizing_mode="fixed", margin=(15,0,20,15),width=650)

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
            """,sizing_mode="fixed", margin=(15,0,20,15),width=650)

            page_1_2=pn.Column(title_2,b,defi_2,margin=(0,0,0,10))

            fomu1=pn.Row(page_1,page_1_2)
####### 3페이지 
            title_3 = pn.pane.Markdown("""
            -------------------                           
            * **수환경 인체 간접노출량 산정식**
            """, width=500)

            c = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & Exposure \text { } (ug/kg \cdot day) = \frac {Ingestion \text { } Amount \text { } (g/day \text { or } L/day) \times Conc. \text { } (ug/g \text { or }ug/L)}{BodyWeight\text { } (kg)} & \end{aligned}$
            """, style={'font-size': '18pt'})

            fomu2=pn.Column(title_3,c)
########3-2 page
            title_4 = pn.pane.Markdown("""
            -------------------                           
            * **어패류 중 물질농도 계산식**
            """, width=500)

            d = pn.pane.LaTeX(r"""                  
            $\begin{aligned} & BAF = \frac {C_B}{C_W}  = (1-L_B)+ \frac {k_1 \cdot \phi + (k_D \cdot \beta \cdot \tau \cdot \phi \cdot L_D \cdot K_{ow})} {k_2 + k_E + k_G + k_M} & \end{aligned}$
            """, style={'font-size': '20pt'})

            e = pn.pane.LaTeX(r"""                  
            $\begin{aligned} \phi = \frac {1}{1+{\chi}_{POC} \cdot 0.35 \cdot K_{ow} + {\chi}_{DOC} \cdot 1 \cdot 0.35 \cdot K_{ow}} \end{aligned}$
            """, style={'font-size': '20pt'})

            f = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_1 = \frac {1}{(0.01+ \frac{1}{K_{ow}})\cdot W^{0.4}}  \end{aligned}$
                    
            """, style={'font-size': '20pt'})

            g = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_2 = \frac {k_1}{L_B \cdot K_{ow}}  \end{aligned}$
                    
            """, style={'font-size': '20pt'})

            h = pn.pane.LaTeX(r"""                  
                            
            $\begin{aligned} k_D = \frac {0.02 \cdot W^{-0.15} \cdot e^{(0.06 \cdot T)}}{5.1 \cdot 10^{-8} \cdot K_{ow} +2}  \end{aligned}$
                    
            """, style={'font-size': '20pt'})

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

            """,sizing_mode="fixed", margin=(15,0,20,15),width=650)

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
            """,sizing_mode="fixed", margin=(15,0,20,15),width=350)
            # fomu3_sub=pn.Row(defi_3,defi_4)
            fomu3=pn.Column(title_4,d,e,f,g,pn.Row(defi_3,defi_4))
#######################################

            @pn.depends(x=radio_group3.param.value)
            def main_s(x):
                if x =='수환경 예측환경농도 입력정보':
                    tab=pn.Column(flow_1,fomu1,mark2,table_a,table4,mark2,table_b,mark2,table_c_a)
                elif x =='수생태 예측환경농도':
                    tab=pn.Column(mark5,mark8,table,pie_part,mark2,table3,mark3,mark2,page_2_3_mark,select_area,mock2,mock4,pn.Row(pec_fig,map_s),pn.Row(pec_table,pnec_ratio),mark2)
                    # tab=pn.Column(mark5,mark8,table,pie_part,mark2,table3,mark3,mark2,page_2_3_mark,select_area,mock2,pn.Row(pec_fig,map_s),pn.Row(pec_table,pnec_ratio),mark2)
                elif x =='수환경 인체 간접노출평가 입력정보':
                    tab=pn.Column(flow_3,fomu2,mark2,figure_4,fomu3,mark2,mock3,mark2,mark9,weight_plot_pn,mark2,dis_water_plot_pn,dis_fish_plot_pn)
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

            # def create_Distribution_exposure_product_table() :
            #     text = """  
            #     """
            #     data = dict(
            #         col = [5,50,75,90,95,99],
            #         dates=p2_table_data,
            #     )
            #     source = bokeh.models.ColumnDataSource(data)

            #     columns = [
            #         bokeh.models.widgets.TableColumn(
            #             field="col", title="분위",
            #         ),
            #         bokeh.models.widgets.TableColumn(
            #             field="dates", title="노출량 (mg/kg/day)",
            #         ),
            #     ]
            #     data_table_all = bokeh.models.widgets.DataTable(
            #         source=source, columns=columns, width=390, height=390, sizing_mode="fixed"
            #     )
            #     data = dict(
            #         col = [5,50,75,90,95,99],
            #         dates=p3_table_data,
            #     )
            #     source = bokeh.models.ColumnDataSource(data)

            #     columns = [
            #         bokeh.models.widgets.TableColumn(
            #             field="col", title="분위",
            #         ),
            #         bokeh.models.widgets.TableColumn(
            #             field="dates", title="노출량 (mg/kg/day)",
            #         ),
            #     ]
            #     data_table_m = bokeh.models.widgets.DataTable(
            #         source=source, columns=columns, width=200, height=390, sizing_mode="fixed"
            #     )
            #     data = dict(
            #         col = [5,50,75,90,95,99],
            #         dates=p4_table_data,
            #     )
            #     source = bokeh.models.ColumnDataSource(data)

            #     columns = [
            #         bokeh.models.widgets.TableColumn(
            #             field="col", title="분위",
            #         ),
            #         bokeh.models.widgets.TableColumn(
            #             field="dates", title="노출량 (mg/kg/day)",
            #         ),
            #     ]
            #     data_table_w = bokeh.models.widgets.DataTable(
            #         source=source, columns=columns, width=200, height=390, sizing_mode="fixed"
            #     )

            #     grid = bokeh.layouts.grid(
            #         children=[
            #             _markdown(text),
            #             [data_table_all],
            #             [data_table_m],
            #             [data_table_w],
            #         ],
            #     )
            #     return grid


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
mark_input=("### * 계산 버튼을 누르기전에 <br> 파일을 업로드 했는지 확인해주세요")

template = pn.template.FastListTemplate(
    site="EHR&C", title="살생물제 및 생활화학제품 사용에 따른 생태 및 인체노출량 산정 프로그램" ,
    
    sidebar=[selcet_input,mark,search_chemi,mark_input,mark2,select_cami],
    main=[calculate_A_batch],
    font="NanumBarunGothic",
    header_background='#2edd2b',
    background_color="#ffffff",
    theme_toggle=False,
    # accent_base_color="#fffcfc",
    neutral_color="#ffffff"
    # theme="dark"
    # shadow=False
    # main=[area]
)
template.sidebar_width=600
template.servable()
