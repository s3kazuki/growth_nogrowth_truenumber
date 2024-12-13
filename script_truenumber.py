import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize

#データの読み込み
@st.cache(allow_output_mutation=True)
def load_data():
    d = pd.read_csv("data_demo.csv").iloc[:, 1:]
    return d
d = load_data()

# 表示用データの加工
d_out = d.copy()
d_out['N0'] = round(d_out['N0'], 1)
d_out['pH'] = round(d_out['pH'], 1)
d_out['aw'] = round(d_out['aw'], 2)


#データの確認
st.title("Growth/no growth boundary")
st.sidebar.title("Input data set(rounded)")
st.sidebar.write(d_out)
st.sidebar.title("Input data set(real)")
st.sidebar.write(d)
st.sidebar.title("Summary of data")
st.sidebar.write(d_out.describe())

option = st.selectbox(
    'How would you like to change?',
     ('pH', 'Water activity', 'Temperature'))

#全要因総当たりモデル
@st.cache(allow_output_mutation=True)
def probability(params, data):
    """
    確率 p を計算する関数
    params: [β0, β1, β2, β3, β4, β5, β6, n] （推定するパラメータ）
    Temp, pH, Salt: モデルの入力データ
    """
    Temp, pH, aw, n = data['Temp'], data['pH'], data['aw'], data['N0']
    β0, β1, β2, β3, β4, β5, β6 = params
    # 線形予測子 η の計算
    eta = (
        β0 +
        β1 * Temp +
        β2 * pH +
        β3 * aw +
        β4 * (Temp * pH) +
        β5 * (Temp * aw) +
        β6 * (pH * aw)
    )
    # 改良された確率計算式
    base_prob = 1 / (1 + np.exp(-eta))  # ロジスティック関数
    p = 1 - (1 - base_prob) ** (10**n)
    return p

# 尤度関数の定義
def negative_log_likelihood(params, data):
    """
    負の対数尤度を計算する関数（最適化対象）
    params: [β0, β1, β2, β3, β4, β5, β6]
    Temp, pH, Salt, n: 説明変数
    gng: 応答変数（観測データ, 0 または 1）
    """
    #gngの定義
    gng = data['gng'] 
    # 確率 p を計算
    p = probability(params, data)
    # 負の対数尤度の計算
    epsilon = 1e-10  # 数値的安定性のための微小値
    log_likelihood = np.sum(gng * np.log(p + epsilon) + (1 - gng) * np.log(1 - p + epsilon))
    return -log_likelihood  # 負の対数尤度を返す

# 初期パラメータの設定
initial_params = [0.1] * 7  # [β0, β1, ..., β6]

# 最適化の実行
result = minimize(
    negative_log_likelihood,  # 最適化する対象の関数
    initial_params,  # 初期パラメータ
    args=d,  # 関数に渡す追加引数
    method='L-BFGS-B',  # 最適化手法
    bounds=[(-10, 10)] * 7   # パラメータの範囲
)

# 最適化結果
fitted_params = result.x

if option == 'pH':
	#pHの入力
	pH_input = st.slider("pH", 3.8, 6.4, 6.0, step=0.1)
	n0_input = st.slider("N$_0$", 1.0, 6.8, 2.6, step=0.2)
	
	#pHの値によるグラフの作成
	dpH = d_out[d_out['pH'] == pH_input]
	dpH = dpH[dpH['N0'] == n0_input]
	dpH_g = dpH[dpH['gng'] == 1]
	dpH_ng = dpH[dpH['gng'] == 0]
	
	
	#x軸は温度
	x = np.arange(3, 35, 0.01)
	#y軸は水分活性
	y = np.arange(0.88, 1.00, 0.0001)
	#z軸はph (%)
	z = np.arange(3.5, 7, 0.01)
	
	#P = 1-(1-f(x)) **n # シングルヒットモデルへのロジスティック回帰の当てはめ
	def func(x,y):
		eta = (
		    fitted_params[0] +
		    fitted_params[1] * x +
		    fitted_params[2] * pH_input +
		    fitted_params[3] * y +
		    fitted_params[4] * (x * pH_input) +
		    fitted_params[5] * (x * y) +
		    fitted_params[6] * (pH_input * y)
		)
		# 改良された確率計算式
		base_prob = 1 / (1 + np.exp(-eta))  # ロジスティック関数
		P = 1 - (1 - base_prob) ** (10**n0_input)
		return P
	
	X, Y = np.meshgrid(x, y)
	Z = func(X, Y)
	
	fig, ax = plt.subplots()
	im = ax.imshow(Z, extent=[3, 35, 0.88, 1.00], origin='lower', aspect='auto')
	plt.colorbar(im, ax=ax, label='Probability of growth')
	ax.set_xlabel('Temperature (°C)', fontsize=12)
	ax.set_ylabel('a$_w$ (–)', fontsize=12)
	ax.tick_params(axis='both', which='major', labelsize=12)
	cs=ax.contour(x, y, Z, levels=[0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9], linewidths=1, colors=['k', 'k', 'k'])
	ax.scatter(dpH_ng['Temp'], dpH_ng['aw'], marker='s', s=50, color='black', label='No growth')
	ax.scatter(dpH_g['Temp'], dpH_g['aw'], marker='o', s=50, color='white', edgecolors='black', label='Growth')
	ax.clabel(cs)
	ax.legend(fontsize=12)
	st.pyplot(fig)

if option == 'Water activity':
	#awの入力
	aw_input = st.slider("a$_w$", 0.890, 1.000, 0.970, step=0.005)
	n0_input = st.slider("N$_0$", 1.0, 6.8, 2.6, step=0.2)
	
	#Saltの値によるグラフの作成
	daw = d_out[d_out['aw'] == aw_input]
	daw = daw[daw['N0'] == n0_input]
	daw_g = daw[daw['gng'] == 1]
	daw_ng = daw[daw['gng'] == 0]
	
	
	#x軸は温度
	x = np.arange(3, 35, 0.01)
	#y軸はph
	y = np.arange(3.5, 7, 0.01)
	#z軸は水分活性
	z = np.arange(0.88, 1.00, 0.0001)
	
	#P = 1-(1-f(x)) **n # シングルヒットモデルへのロジスティック回帰の当てはめ
	def func(x,y):
		eta = (
		    fitted_params[0] +
		    fitted_params[1] * x +
		    fitted_params[2] * y +
		    fitted_params[3] * aw_input +
		    fitted_params[4] * (x * y) +
		    fitted_params[5] * (x * aw_input) +
		    fitted_params[6] * (y * aw_input)
		)
		# 改良された確率計算式
		base_prob = 1 / (1 + np.exp(-eta))  # ロジスティック関数
		P = 1 - (1 - base_prob) ** (10**n0_input)
		return P
	
	X, Y = np.meshgrid(x, y)
	Z = func(X, Y)
	
	fig, ax = plt.subplots()
	im = ax.imshow(Z, extent=[3, 35, 3.5, 7], origin='lower', aspect='auto')
	plt.colorbar(im, ax=ax, label='Probability of growth')
	ax.set_xlabel('Temperature (°C)', fontsize=12)
	ax.set_ylabel('pH', fontsize=12)
	ax.tick_params(axis='both', which='major', labelsize=12)
	cs=ax.contour(x, y, Z, levels=[0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9], linewidths=1, colors=['k', 'k', 'k'])
	ax.scatter(daw_ng['Temp'], daw_ng['pH'], marker='s', s=50, color='black', label='No growth')
	ax.scatter(daw_g['Temp'], daw_g['pH'], marker='o', s=50, color='white', edgecolors='black', label='Growth')
	
	ax.legend(fontsize=12)
	ax.clabel(cs)
	st.pyplot(fig)

if option == 'Temperature':
	#温度の入力
	temperature_input = st.slider("Temperature", 4.0, 30.0, 20.0, step=0.1)
	n0_input = st.slider("N$_0$", 1.0, 6.8, 2.6, step=0.2)
	
	#温度の値によるグラフの作成
	dTemperature = d_out[d_out['Temp'] == temperature_input]
	dTemperature = dTemperature[dTemperature['N0'] == n0_input]
	dTemperature_g = dTemperature[dTemperature['gng'] == 1]
	dTemperature_ng = dTemperature[dTemperature['gng'] == 0]
	
	#x軸はpH
	x = np.arange(3.5, 7, 0.01)
	#y軸は水分活性
	y = np.arange(0.88, 1.00, 0.0001)
	#z軸は温度
	z = np.arange(3, 35, 0.01)
	
	#P = 1-(1-f(x)) **n # シングルヒットモデルへのロジスティック回帰の当てはめ
	def func(x,y):
		eta = (
		    fitted_params[0] +
		    fitted_params[1] * temperature_input +
		    fitted_params[2] * x +
		    fitted_params[3] * y +
		    fitted_params[4] * (temperature_input * x) +
		    fitted_params[5] * (temperature_input * y) +
		    fitted_params[6] * (x * y)
		)
		# 改良された確率計算式
		base_prob = 1 / (1 + np.exp(-eta))  # ロジスティック関数
		P = 1 - (1 - base_prob) ** (10**n0_input)
		return P
	
	X, Y = np.meshgrid(x, y)
	Z = func(X, Y)
	
	fig, ax = plt.subplots()
	im = ax.imshow(Z, extent=[3.5, 7, 0.88, 1.00], origin='lower', aspect='auto')
	plt.colorbar(im, ax=ax, label='Probability of growth')
	ax.set_xlabel('pH', fontsize=12)
	ax.set_ylabel('a$_w$ (–)', fontsize=12)
	ax.tick_params(axis='both', which='major', labelsize=12)
	cs=ax.contour(x, y, Z, levels=[0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9], linewidths=1, colors=['k', 'k', 'k'])
	ax.scatter(dTemperature_ng['pH'], dTemperature_ng['aw'], marker='s', s=50, color='black', label='No growth')
	ax.scatter(dTemperature_g['pH'], dTemperature_g['aw'], marker='o', s=50, color='white', edgecolors='black', label='Growth')
	ax.clabel(cs)
	ax.legend(fontsize=12)
	st.pyplot(fig)
	
	
	
import base64
from io import BytesIO
pdf_bytes = BytesIO()
plt.savefig(pdf_bytes, format='pdf')
plt.close(fig)
    
# ダウンロード用のリンクを生成
pdf_data = pdf_bytes.getvalue()
b64 = base64.b64encode(pdf_data).decode()
href = f'<a href="data:application/pdf;base64,{b64}" download="output.pdf">Download Figure</a>'
st.markdown(href, unsafe_allow_html=True)


st.write("Other app: https://scrapbox.io/kentokoyama/web_application")
st.write("Contact: kento.koyama.123@gmail.com")
