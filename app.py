from flask import Flask, render_template, request, redirect, make_response
import pandas as pd
from sklearn.decomposition import PCA  # 主成分分析器
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# 入力とチェック
def check_request(data_request):
    csv_file = data_request.files['csv']
    component_number = data_request.values['num_member']

    if csv_file.content_type != 'text/csv':
        print("ファイルがありません")
        return False, None, None

    if component_number == "":
        print('数字がありません')
        return False, None, None

    # 入力されている場合

    # 解析データの形式

    # 元データ読み込み
    header_flg = 0  # ヘッダー（列名）は必要：　有り 0）
    index_flg = 0  # index（Sample名)：　有り 0
    raw_score = pd.read_csv(csv_file, header=header_flg, index_col=index_flg)
    return True, raw_score, int(component_number)


# メインの処理
@app.route('/', methods=["GET", "POST"])
def main():
    # 通常のページへのアクセス対応
    if request.method == 'GET':
        return render_template('initial.html')

    # フォーム送信（POST)があった際の対応
    data_request = request
    input_check, raw_score, component_number = check_request(data_request)  # フォームから入力

    # 入力が不正の場合は元にもどる
    if input_check == False:
        return redirect(request.url)

    # 入力が正しい場合は解析処理

    # PCA処理

    global Contribution_Ratio  # 寄与率( %): pca.explained_variance_ratio_
    global Eigen_Vector  # 固有ベクトル(主成分の方向): pca.components_
    global Principal_Component_Score  # 主成分スコア(データの主成分方向での値): pca.transform(X)
    global std_recovered_score  # 逆変換
    global recovered_raw_score  # 逆変換元データ

    Contribution_Ratio, Eigen_Vector, Principal_Component_Score, std_recovered_score, recovered_raw_score = \
        pca_analysis(raw_score, component_number)


    return render_template('result.html')


'''
寄与率(%): Contribution_Ratio  pca.explained_variance_ratio_        
固有ベクトル: Eigen_Vector    pca.components_         
主成分スコア:Principal_Component_Score  pca.transform(X)
逆変換", std_recovered_score
逆変換元データ", recovered_raw_score
'''


def pca_analysis(raw_score, component_number):
    # 標準化
    sc = StandardScaler().fit(raw_score)
    std_score = pd.DataFrame(sc.transform(raw_score))  # 標準化データ

    # 主成分分析の実行
    pca = PCA(n_components=component_number)
    pca.fit(std_score)

    # 各種データ出力
    Contribution_Ratio = pd.DataFrame(pca.explained_variance_ratio_)
    Eigen_Vector = pd.DataFrame(pca.components_)
    Principal_Component_Score = pd.DataFrame(pca.transform(std_score))

    # 逆変換
    list_std_recovered_score = pca.inverse_transform(Principal_Component_Score)
    std_recovered_score = pd.DataFrame(list_std_recovered_score, columns=raw_score.columns)

    # 標準化データを元にもどす
    recovered_raw_score = pd.DataFrame(sc.inverse_transform(std_recovered_score),
                                       index=raw_score.index, columns=raw_score.columns)

    return Contribution_Ratio, Eigen_Vector, Principal_Component_Score, std_recovered_score, recovered_raw_score


# 寄与率(%): Contribution_Ratio  pca.explained_variance_ratio_
# 固有ベクトル: Eigen_Vector    pca.components_
# 主成分スコア:Principal_Component_Score  pca.transform(X)
# 逆変換", std_recovered_score
# 逆変換元データ", recovered_raw_score

# 寄与率(%)
@app.route('/Contribution_Ratio', methods=['GET', 'POST'])
def Contribution_Ratio():
    csv = Contribution_Ratio.to_csv(index=False)
    response = make_response()
    response.data = csv
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment;filename=Contribution_Ratio.csv'
    return response


# 固有ベクトル
@app.route('/Eigen_Vector', methods=['GET', 'POST'])
def Eigen_Vector():
    csv = Eigen_Vector.to_csv(index=False)
    response = make_response()
    response.data = csv
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment;filename=Eigen_Vector.csv'
    return response


# 主成分スコア
@app.route('/Principal_Component_Score', methods=['GET', 'POST'])
def Principal_Component_Score():
    csv = Principal_Component_Score.to_csv(index=False)
    response = make_response()
    response.data = csv
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment;filename=Principal_Component_Score.csv'
    return response


# 逆変換
@app.route('/std_recovered_score', methods=['GET', 'POST'])
def std_recovered_score():
    csv = std_recovered_score.to_csv(index=False)
    response = make_response()
    response.data = csv
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment;filename=std_recovered_score.csv'
    return response


# 逆変換元データ
@app.route('/recovered_raw_score', methods=['GET', 'POST'])
def recovered_raw_score():
    csv = recovered_raw_score.to_csv(index=False)
    response = make_response()
    response.data = csv
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment;filename=recovered_raw_score.csv'
    return response


## 実行
if __name__ == "__main__":
    app.run(debug=True)
