from flask import Flask, request, jsonify, render_template
from math import *
import math as m
from scipy.stats import norm
from fractions import Fraction
from sympy import *
import sympy as sp
import re
import os

app = Flask(__name__)

_custom_func_expr = 'x'

# =========================== 工具函数 ===========================
def fix_expression(expr):
    """自动补全缺失的乘号，兼容用户输入 3x、2sin(x) 等写法"""
    if not expr:
        return 'x'
    # 规则1：数字后紧跟字母/括号 → 加*（如 3x → 3*x, 5(sin) →5*(sin)）
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
    # 规则2：括号后紧跟数字/字母 → 加*（如 (x)3 → (x)*3, (x)y → (x)*y）
    expr = re.sub(r'([)])\s*(\d|\w)', r'\1*\2', expr)
    # 规则3：移除多余空格
    expr = expr.replace(' ', '')
    return expr

# =========================== 保留原有函数和逻辑 ===========================
def custom_func(x):
    try:
        x_sym = sp.Symbol('x')
        fixed_expr = fix_expression(_custom_func_expr)
        expr = sp.sympify(_custom_func_expr)
        return expr.subs(x_sym, x)
    except Exception as e:
        print(f"自定義函數解析失敗: {e}")
        return x


def func(x):
    x_sym = sp.Float(x) if isinstance(x, (int, float)) else x
    return custom_func(x_sym)

def is_real(n):
    if isinstance(n, (int, float)):
        return not (m.isnan(n) or m.isinf(n))
    elif hasattr(n, 'is_real') and hasattr(n, 'is_finite'):
        return n.is_real and n.is_finite
    else:
        return False

# 封装所有计算逻辑为函数（保留原变量名和逻辑）
def calculate(choice, params):
    global _custom_func_expr  # 声明使用全局变量

    # 在函數開頭，優先讀取前端傳來的 custom_func 表達式
    if 'custom_func' in params and params['custom_func'].strip():
        _custom_func_expr = fix_expression(params['custom_func'].strip())
        print(f"成功更新自定义函数: {_custom_func_expr}")  # 调试用，可删除
    else:
        _custom_func_expr = 'x'

    result = []  # 用列表存储输出结果，替代原print

    try:
        choice = int(choice)
        params = {k: v for k, v in params.items()}  # 转换参数为字典

        # 辅助打印函数（替代原print）
        def print_result(msg):
            result.append(msg)

        # -------------------------------------------------------------------------------------------
        if choice == 0:
            print_result(f"{'---' * 20}\n離開程式\n{'---' * 20}")

        # -------------------------------------------------------------------------------------------
        elif choice == 1:  # 二項式展開
            print_result(
                f"{'---' * 20}\n二項式展開係數 (A+Bx)^n = A^n + nC1 * A^(n-1)*(Bx) + nC2 * A^(n-2)*(Bx)^2+...\n{'---' * 20}")

            A = float(params['A'])
            B = float(params['B'])
            n = int(params['n'])

            print_result(f"{'---' * 20}\n將會展開({A} _  + {B} _ )^{n}\n{'---' * 20}")
            for i in range(n, -1, -1):
                print_result(f"第{n - i + 1}項: {m.comb(n, i) * A ** i * B ** (n - i)}")
            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 2:  # e的展開
            print_result(f"{'---' * 20}\ne的展開 = 1+x+x²/2!+x³/3!+....\n{'---' * 20}")

            k = float(params['k'])
            n = int(params['n'])

            if abs(k) <= 1e-9:
                print_result(f"{'---' * 20}\n第1項係數 : 1")
            else:
                print_result(f"現展開e^({k}x)")
                for i in range(n):
                    print_result(
                        f"第 {i + 1} 項係數:{k ** i / factorial(i)} (分數形式:{Fraction(k ** i / m.factorial(i)).limit_denominator()})")
            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 3:  # 伯努利分布
            print_result(f"{'---' * 20}\n伯努利分布 X~Ber(p) P(X=k) = p^k * (1-p)^(1-k) (k=0,1)\n{'---' * 20}")
            p = float(params['p'])
            k = int(params['k'])

            print_result(f"{'---' * 20}\n期望值E(X): {p}  方差Var(X): {p * (1 - p)}\n{'---' * 20}")
            print_result(f"P(X=0) = {1 - p:.4f}\nP(X=1) = {p:.4f}\n{'---' * 20}")

        # -------------------------------------------------------------------------------------------
        elif choice == 4:  # 泊松分布
            print_result(f"{'---' * 20}\n泊松分布 X~P(λ) P(X=k)=e^(-λ)*(λ^k)/k! (λ>=0)\n{'---' * 20}")

            # 1. 安全读取参数（空值/不存在默认转0）
            mu_str = params.get('mu', '0') or '0'
            n_str = params.get('n', '0') or '0'
            k_str = params.get('k', '0') or '0'

            is_valid = True
            error_msg = ""
            mu , n ,k = 0.0 , 0 , 0

            try:
                mu = float(mu_str)
                n = int(float(n_str))
                k = int(float(k_str))
            except ValueError:
                error_msg = "❌ 輸入錯誤：請輸入有效的數字！"
                is_valid = False

            if is_valid:
                if mu < 0:
                    error_msg = "❌ 錯誤：泊松分布的平均值 λ 必須 ≥ 0！"
                    is_valid = False
                elif n < 0 or k < 0:
                    error_msg = "❌ 錯誤：發生次數 n/k 必須 ≥ 0！"
                    is_valid = False
                elif n > k:
                    error_msg = f"❌ 錯誤：起始次數 n({n}) 不能大於結束次數 k({k})！"
                    is_valid = False

            if not is_valid:
                print_result(error_msg)
                print_result('---' * 20)
            else:
                print_result(f"{'---' * 20}\n期望值E(X): {mu}  方差Var(X): {mu}\n{'---' * 20}\n"
                             f"請輸入由 n 到 k 的兩個值(所求事件發生次數) (若所求發生次數為單次，重複輸入兩次就可以)")

                total = 0.0
                for i in range(n, k + 1):
                    ANS = (mu ** i) * m.exp(-mu) / m.factorial(i)
                    print_result(f"P(X={i})={ANS} (準確至四位小數：{ANS:.4f}) ")
                    total += ANS

                print_result(f"{'---' * 20}\n{n}~{k}次成功的概率：{float(total)}\n"
                             f"準確至四位小數：{float(total):.4f}")

        # -------------------------------------------------------------------------------------------
        elif choice == 5:  # 二項概率分布
            print_result(f"{'---' * 20}\n二項概率分布 X~B(n,p) P(X=k) = nCk * p^k * (1-p)^(n-k)\n{'---' * 20}")

            n_str = params.get('n', '0') or '0'
            p_str = params.get('p', '0') or '0'
            k_str = params.get('k', '0') or '0'

            is_valid = True
            error_msg = ""
            n, p, k = 0, 0.0, 0
            try:
                n = int(float(n_str))
                p = float(p_str)
                k = int(float(k_str))
            except ValueError:
                error_msg = "❌ 輸入錯誤：請輸入有效的數字！"
                is_valid = False

            if is_valid:
                if n < 1:
                    error_msg = f"❌ 錯誤：總試驗次數 n({n}) 必須 ≥ 1！"
                    is_valid = False
                elif p < 0 or p > 1:
                    error_msg = f"❌ 錯誤：成功概率 p({p}) 必須在 0~1 之間！"
                    is_valid = False
                elif k < 0 or k > n:
                    error_msg = f"❌ 錯誤：成功次數 k({k}) 必須 ≥0 且 ≤ 總試驗次數 n({n})！"
                    is_valid = False

            if not is_valid:
                print_result(error_msg)
                print_result('---' * 20)

            else:
                e_x = n * p
                var_x = n * p * (1 - p)
                print_result(f"{'---' * 20}\n期望值E(X) : {e_x}\n方差Var(X) : {var_x}\n{'---' * 20}")
                print_result(f"成功概率 : {p}\n失敗概率 : {1 - p}")

                total = 0.0
                for i in range(k + 1):
                    ANS = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                    print_result(f"P(X={i}): {ANS} (準確至四位小數: {ANS:.4f})")
                    total += ANS

                exact_prob = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

                print_result(f"{'---' * 20}\n0~{k}次成功的概率的和 P( 0 < X < {k}) = {total} (準確至四位小數: {total:.4f})")
                print_result(f"恰好{k}次成功的概率P( X = {k}) = {exact_prob} (準確至四位小數: {exact_prob:.4f})")

        # -------------------------------------------------------------------------------------------

        elif choice == 6:  # 標準正態分布
            print_result(f"{'---' * 20}\n常態分布/正態分布/正向計算/反查正態分布\n{'---' * 20}")

            # 1. 安全读取所有参数（空值/不存在默认转0）
            mu_str = params.get('µ', '0') or '0'
            mu_2_str = params.get('σ', '0') or '0'
            choices = params.get('choices', '0')
            B_str = params.get('B', '0') or '0'
            A_str = params.get('A', '0') or '0'
            p_str = params.get('p', '0') or '0'

            is_valid = True
            error_msg = ""
            mu, mu_2 = 0.0, 0.0

            try:
                mu = float(mu_str)
                mu_2 = float(mu_2_str)
            except ValueError:
                error_msg = "❌ 輸入錯誤：平均值 µ 或 標準差 σ 必須是數字！"
                is_valid = False

            if is_valid:
                if mu_2 <= 0:
                    error_msg = f"❌ 錯誤：標準差 σ ({mu_2}) 必須大於 0！"
                    is_valid = False
                # 校验选择项合法性
                elif choices not in ['0', '1']:
                    error_msg = f"❌ 錯誤：選擇項必須是 0(正向計算) 或 1(反查)，當前值：{choices}"
                    is_valid = False

            # 4. 分场景校验参数
            if is_valid:
                if choices == '0':  # 正向计算：需要A、B参数
                    try:
                        B = float(B_str)
                        A = float(A_str)
                    except ValueError:
                        error_msg = "❌ 輸入錯誤：區間值 A/B 必須是數字！"
                        is_valid = False
                else:  # 反查：需要概率p参数
                    try:
                        p = float(p_str)
                        # 概率必须在0~1之间
                        if p < 0 or p > 1:
                            error_msg = f"❌ 錯誤：概率 p({p}) 必須在 0~1 之間！"
                            is_valid = False
                    except ValueError:
                        error_msg = "❌ 輸入錯誤：概率 p 必須是數字！"
                        is_valid = False

            # 5. 无效参数：输出错误提示
            if not is_valid:
                print_result(error_msg)
                print_result('---' * 20)
            # 6. 有效参数：执行计算逻辑
            else:
                print_result(f"{'---' * 20}\n***X~N({mu},{mu_2}²)***\n{'---' * 20}")

                if choices == '0':  # 正向计算：P(X≤B)、P(A≤X≤B)等
                    B = float(B_str)
                    A = float(A_str)

                    # 计算P(0≤X≤B) 或 P(B≤X≤0)
                    print_result('---' * 20)
                    if B > 1e-9:  # B为正数
                        p_0_to_B = norm.cdf(B, loc=mu, scale=mu_2) - norm.cdf(0, loc=mu, scale=mu_2)
                        print_result(f"P(0<=X<={B}): {p_0_to_B} (準確至三位小數：{p_0_to_B:.3f})")
                    elif B < -1e-9:  # B为负数（避免浮点精度问题）
                        p_B_to_0 = norm.cdf(0, loc=mu, scale=mu_2) - norm.cdf(B, loc=mu, scale=mu_2)
                        print_result(f"P({B}<=X<=0): {p_B_to_0} (準確至三位小數：{p_B_to_0:.3f})")
                    else:  # B≈0
                        print_result(f"P(0<=X<=0): 0.0 (準確至三位小數：0.000)")

                    # 计算P(X<B) 和 P(X>B)
                    p_x_lt_B = norm.cdf(B, loc=mu, scale=mu_2)
                    p_x_gt_B = 1 - p_x_lt_B
                    print_result(f"P(X<{B}): {p_x_lt_B} (準確至三位小數：{p_x_lt_B:.3f})")
                    print_result(f"P(X>{B}): {p_x_gt_B} (準確至三位小數：{p_x_gt_B:.3f})")
                    print_result('---' * 20)

                    # 修复A/B顺序交换逻辑（原代码有变量覆盖bug）
                    A_final = min(A, B)
                    B_final = max(A, B)
                    p_A_to_B = norm.cdf(B_final, loc=mu, scale=mu_2) - norm.cdf(A_final, loc=mu, scale=mu_2)
                    print_result(
                        f"{'---' * 20}\nP({A_final}<=X<={B_final})={p_A_to_B}(準確至三位小數：{p_A_to_B:.3f})\n{'---' * 20}")

                else:
                    p = float(p_str)

                    if p < 0.5 + 1e-9:
                        cdf_0 = norm.cdf(0, loc=mu, scale=mu_2)
                        target_cdf = cdf_0 + p
                        # 防错：target_cdf不能超出0~1范围
                        target_cdf = max(0.0, min(1.0, target_cdf))
                        B_0_to_p = norm.ppf(target_cdf, loc=mu, scale=mu_2)
                        print_result(f"當P(0<X<B)={p}時: B={B_0_to_p}(準確至三位小數：{B_0_to_p:.3f})\n{'---' * 20}")

                    # 计算P(X<B)=p 时的B值
                    # 防错：p不能超出0~1范围（已在校验阶段处理，此处双重保障）
                    p_clamped = max(0.0, min(1.0, p))
                    B_p = norm.ppf(p_clamped, loc=mu, scale=mu_2)
                    print_result(f"當P(X<B)={p}時: B={B_p}(準確至三位小數：{B_p:.3f})\n{'---' * 20}")

        # -------------------------------------------------------------------------------------------
        elif choice == 7:  # 直線方程/分點座標等
            print_result(
                f"{'---' * 20}\n兩點直線方程 y = mx+c，內分點座標，垂直分點方程，三點求圓，四心座標，內接圓方程，三角形面積(海倫公式)，三隻角的度數\n{'---' * 20}")

            A_x = float(params['A_x'])
            A_y = float(params['A_y'])
            B_x = float(params['B_x'])
            B_y = float(params['B_y'])
            C_x_R = params.get('C_x', '')
            C_y_R = params.get('C_y', '')
            C_x = float(C_x_R) if C_x_R else None
            C_y = float(C_y_R) if C_y_R else None

            if C_x is None and C_y is None:
                if A_x == B_x:
                    print_result(f"兩點在垂直直線上，斜率不存在，直線方程為 x = {A_x} 距離為 {abs(A_y - B_y)}")

                    C_R = params.get('C', '')
                    D_R = params.get('D', '')
                    if C_R and D_R:
                        C = float(C_R)
                        D = float(D_R)
                        x_1 = A_x
                        y_1 = (C * B_y + D * A_y) / (C + D)
                        print_result(f"內分點座標 x = {x_1} (分數形式:{Fraction(x_1).limit_denominator()})\n"
                                     f"內分點座標 y = {y_1} (分數形式:{Fraction(y_1).limit_denominator()})\n"
                                     f"通過該分點且與 A B 垂直方程為 y = {y_1}\n{'---' * 20}")
                else:
                    M = (A_y - B_y) / (A_x - B_x)
                    y = -M * A_x + A_y
                    n = "+" if y >= 0 else "-"

                    if M == 0:
                        print_result(
                            f"{'---' * 20}\n兩點斜率:0\n直線方程/y截距 = {y} (分數形式 : {Fraction(y).limit_denominator()})\n{'---' * 20}")
                    else:
                        print_result(f"{'---' * 20}\n"
                                     f"兩點斜率= {M} (分數形式 : {Fraction(M).limit_denominator()})\ny截距 = {y} (分數形式 : {Fraction(y).limit_denominator()})\n"
                                     f"直線方程:y = {Fraction(M).limit_denominator()}x {n} {abs(Fraction(y).limit_denominator())}\n"
                                     f"{'---' * 20}")

                    C_R = params.get('C', '')
                    D_R = params.get('D', '')
                    if C_R and D_R:
                        C = float(C_R)
                        D = float(D_R)
                        x_1 = (C * B_x + D * A_x) / (C + D)
                        y_1 = (C * B_y + D * A_y) / (C + D)
                        M = Fraction(M).limit_denominator()
                        intercept = y_1 - (-1 / M) * x_1
                        y_2 = Fraction(intercept).limit_denominator()
                        n = "+" if intercept >= 0 else "-"
                        print_result(f"內分點座標 x = {x_1} (分數形式:{Fraction(x_1).limit_denominator()})\n"
                                     f"內分點座標 y = {y_1} (分數形式:{Fraction(y_1).limit_denominator()})\n"
                                     f"通過該分點且與 A B 垂直方程為 y = {-1 / M}x {n} {abs(y_2)}\n{'---' * 20}")
            else:
                print_result(f"{'---' * 20}\nA( {A_x} , {A_y} )  B( {B_x} , {B_y} )  C( {C_x} , {C_y} )\n{'---' * 20}")

                x_1 = A_x + B_x + C_x
                y_1 = A_y + B_y + C_y
                print_result(f"重心 x 座標 : {x_1 / 3} (分數形式 : {Fraction(x_1 / 3).limit_denominator()})\n"
                             f"重心 y 座標 : {y_1 / 3} (分數形式 : {Fraction(y_1 / 3).limit_denominator()})\n{'---' * 20}")

                a_b = (C_x - B_x) * (A_x - C_x) + (C_y - B_y) * (A_y - C_y)
                a_c = (C_x - B_x) * (B_x - A_x) + (C_y - B_y) * (B_y - A_y)
                b_c = (A_x - C_x) * (B_x - A_x) + (A_y - C_y) * (B_y - A_y)
                a = a_b * a_c
                b = b_c * a_b
                c = a_c * b_c
                sum_abc = a + b + c
                x_2 = (a * A_x + b * B_x + c * C_x) / (a + b + c)
                y_2 = (a * A_y + b * B_y + c * C_y) / (a + b + c)
                print_result(f"垂心 x 座標 : {x_2} (分數形式 : {Fraction(x_2).limit_denominator()})\n"
                             f"垂心 y 座標 : {y_2} (分數形式 : {Fraction(y_2).limit_denominator()})\n{'---' * 20}")

                a = m.hypot(A_x - B_x, A_y - B_y)
                b = m.hypot(C_x - B_x, C_y - B_y)
                c = m.hypot(A_x - C_x, A_y - C_y)
                x_3 = (b * A_x + c * B_x + a * C_x) / (a + b + c)
                y_3 = (b * A_y + c * B_y + a * C_y) / (a + b + c)
                print_result(f"内心 x 座標 : {x_3} (分數形式 : {Fraction(x_3).limit_denominator()})\n"
                             f"内心 y 座標 : {y_3} (分數形式 : {Fraction(y_3).limit_denominator()})\n{'---' * 20}")

                x_4 = (x_1 - x_2 / sum_abc) / 2
                y_4 = (y_1 - y_2 / sum_abc) / 2
                r = m.hypot(x_4 / 2 - A_x, y_4 / 2 - A_y)
                s = (a + b + c) / 2
                Area = (s * (s - a) * (s - b) * (s - c)) ** 0.5

                print_result(f"外心 x 座標 : {x_4} (分數形式 : {Fraction(x_4).limit_denominator()})\n"
                             f"外心 y 座標 : {y_4} (分數形式 : {Fraction(y_4).limit_denominator()})\n{'---' * 20}\n"
                             f"外接圓半徑 r = {r} (√{r ** 2}) (分數形式 : {Fraction(r).limit_denominator()})\n"
                             f"外接圓方程 (x-{Fraction(x_4).limit_denominator()})^2 + (y-{Fraction(y_4).limit_denominator()})^2 = {r ** 2}\n"
                             f"內接圓半徑 r = {Area / s} (√{(Area / s) ** 2}) (分數形式 : {Fraction(Area / s).limit_denominator()})\n"
                             f"內接圓方程 (x-{Fraction(x_3).limit_denominator()})^2 + (y-{Fraction(y_3).limit_denominator()})^2 = {(Area / s) ** 2}\n"
                             f"{'---' * 20}")

                print_result(f"三邊長度 :\n"
                             f"AB = {a} (分數形式 : {Fraction(a).limit_denominator()})\n"
                             f"BC = {b} (分數形式 : {Fraction(b).limit_denominator()})\n"
                             f"AC = {c} (分數形式 : {Fraction(c).limit_denominator()})\n"
                             f"三邊比例 : "
                             f"三角形ABC 的面積 : {Area} (分數形式 : {Fraction(Area).limit_denominator()})\n{'---' * 20}")

                cos_ABC = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
                cos_ACB = (b ** 2 + a ** 2 - c ** 2) / (2 * b * a)
                cos_CAB = (c ** 2 + b ** 2 - a ** 2) / (2 * c * b)

                cos_ABC_fix = max(min(cos_ABC, 1.0), -1.0)
                rad_ABC = m.acos(cos_ABC_fix)
                point_ABC = m.degrees(rad_ABC)

                cos_ACB_fix = max(min(cos_ACB, 1.0), -1.0)
                rad_ACB = m.acos(cos_ACB_fix)
                point_ACB = m.degrees(rad_ACB)

                cos_CAB_fix = max(min(cos_CAB, 1.0), -1.0)
                rad_CAB = m.acos(cos_CAB_fix)
                point_CAB = m.degrees(rad_CAB)

                if 180 - 1e-9 <= (point_ABC + point_CAB + point_ACB) <= 180 + 1e-9:
                    print_result(f"角ABC = {point_ABC:.9f}°")
                    print_result(f"角ACB = {point_ACB:.9f}°")
                    print_result(f"角CAB = {point_CAB:.9f}°")
                else:
                    print_result("未知錯誤，三角形角度總和不等於180度")
                print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 8:  # 圓/拋物線/直線與直線交點
            print_result(f"{'---' * 20}\n圓/拋物線/直線與直線交點\n{'---' * 20}")

            A = float(params.get('A', '0') or '0')
            B = float(params.get('B', '0') or '0')
            C = float(params.get('C', '0') or '0')
            D = float(params.get('D', '0') or '0')

            if abs(A) > 1e-9:
                r = (B / (2 * A)) ** 2 + (C / (2 * A)) ** 2 - D / A
                if r > 1e-9:
                    print_result(
                        f"{'---' * 20}\n圓心座標 ({-B / (2 * A)} , {-C / (2 * A)})\n圓半徑 = {sqrt(r)}\n{'---' * 20}")
                else:
                    print_result(f"{'---' * 20}\n圓方程不是實圓 ! 無法計算交點!\n{'---' * 20}")

            E = float(params.get('E', '0') or '0')
            F = float(params.get('F', '0') or '0')
            G = float(params.get('G', '0') or '0')

            if abs(E) < 1e-9 and abs(F) < 1e-9:
                print_result("直線方程 Ex + Fy = G 至少有一個變量 !")
            else:
                if abs(A) < 1e-9 and abs(B) < 1e-9:  # 直線
                    H_str = params.get('H', '0') or '0'
                    H = float(H_str)
                    if abs(C) < 1e-9 and abs(D) < 1e-9:
                        print_result("直線方程 Cx + Dy = H 至少有一個變量 !")
                    else:
                        if abs(C * F - D * E) < 1e-9:
                            if abs(C * G - E * H) < 1e-9 and abs(D * G - F * H) < 1e-9:
                                print_result(f"{'---' * 20}\n方程有無窮多解（兩直線重合）！\n{'---' * 20}")
                            else:
                                print_result(f"{'---' * 20}\n方程無解（兩直線平行）！\n{'---' * 20}")
                        else:
                            x = (H * F - D * G) / (C * F - D * E)
                            y = (C * G - H * E) / (C * F - D * E)
                            print_result(f"{'---' * 20}\n"
                                         f"交點 ({x} , {y}) "
                                         f"(分數形式:{Fraction(x).limit_denominator()} , "
                                         f"{Fraction(y).limit_denominator()})")
                elif abs(A) < 1e-9:  # 拋物線
                    x_1 = x_2 = y_1 = y_2 = None
                    if abs(E) > 1e-9 and abs(F) > 1e-9:
                        c = D - G / F
                        b = C + E / F
                        if b ** 2 - 4 * B * c > 1e-9:
                            x_1 = (-b + sqrt(b ** 2 - 4 * B * c)) / (2 * B)
                            y_1 = (-E * x_1) / F + G / F
                            x_2 = (-b - sqrt(b ** 2 - 4 * B * c)) / (2 * B)
                            y_2 = (-E * x_2) / F + G / F
                        if abs(b ** 2 - 4 * B * c) < 1e-9:
                            x_1 = -b / (2 * B)
                            y_1 = (E * b + 2 * B * G) / (2 * B * F)

                    if abs(F) < 1e-9 < abs(E):
                        x_1 = G / E
                        y_1 = B * x_1 ** 2 + C * x_1 + D

                    if abs(E) < 1e-9 < abs(F):
                        c = D - G / F
                        if C ** 2 - 4 * B * c > 1e-9:
                            x_1 = (-C + sqrt(C ** 2 - 4 * B * c)) / (2 * B)
                            y_1 = G / F
                            x_2 = (-C - sqrt(C ** 2 - 4 * B * c)) / (2 * B)
                            y_2 = G / F
                        if abs(C ** 2 - 4 * B * c) < 1e-9:
                            x_1 = -C / (2 * B)
                            y_1 = G / F

                    print_result(f"{'---' * 20}")
                    if x_1 is not None and y_1 is not None:
                        print_result(f"交點 ({float(x_1)} , {float(y_1)})\n"
                                     f"(分數形式 : {Fraction(float(x_1)).limit_denominator()} , {Fraction(float(y_1)).limit_denominator()})")
                    if x_2 is not None and y_2 is not None:
                        print_result(f"交點 ({float(x_2)} , {float(y_2)})\n"
                                     f"(分數形式 : {Fraction(float(x_2)).limit_denominator()} , {Fraction(float(y_2)).limit_denominator()})")
                    if all(i is None for i in [x_1, x_2, y_1, y_2]):
                        print_result(f"{'---' * 20}\n沒有交點 !")
                else:  # 圓
                    x_1 = x_2 = y_1 = y_2 = None
                    if abs(F) < 1e-9:
                        c = A * (G / E) ** 2 + B * G / E + D
                        if C ** 2 - 4 * A * c > 1e-9:
                            x_1 = G / E
                            y_1 = (-C + sqrt(C ** 2 - 4 * A * c)) / (2 * A)
                            x_2 = G / E
                            y_2 = (-C - sqrt(C ** 2 - 4 * A * c)) / (2 * A)
                        if abs(C ** 2 - 4 * A * c) < 1e-9:
                            x_1 = G / E
                            y_1 = -C / (2 * A)

                    elif abs(E) < 1e-9:
                        c = A * (G / F) ** 2 + C * G / F + D
                        if B ** 2 - 4 * A * c > 1e-9:
                            x_1 = (-B + sqrt(B ** 2 - 4 * A * c)) / (2 * A)
                            y_1 = G / F
                            x_2 = (-B - sqrt(B ** 2 - 4 * A * c)) / (2 * A)
                            y_2 = G / F
                        if abs(B ** 2 - 4 * A * c) < 1e-9:
                            x_1 = -B / (2 * A)
                            y_1 = G / F

                    else:  # E,F!=0
                        a = A + A * (E / F) ** 2
                        b = B - (2 * A * E * G) / (F ** 2) - E * C / F
                        c = A * (G / F) ** 2 + C * G / F + D
                        if b ** 2 - 4 * a * c > 1e-9:
                            x_1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                            y_1 = (-E * x_1) / F + G / F
                            x_2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                            y_2 = (-E * x_2) / F + G / F
                        if abs(b ** 2 - 4 * a * c) < 1e-9:
                            x_1 = -b / (2 * a)
                            y_1 = (-E * x_1 / F) + G / F

                    print_result(f"{'---' * 20}")
                    if x_1 is not None and y_1 is not None:
                        print_result(f"交點 ({float(x_1)} , {float(y_1)})\n"
                                     f"(分數形式 : {Fraction(float(x_1)).limit_denominator()} , {Fraction(float(y_1)).limit_denominator()})")
                    if x_2 is not None and y_2 is not None:
                        print_result(f"交點 ({float(x_2)} , {float(y_2)})\n"
                                     f"(分數形式 : {Fraction(float(x_2)).limit_denominator()} , {Fraction(float(y_2)).limit_denominator()})")
                    if all(i is None for i in [x_1, x_2, y_1, y_2]):
                        print_result(f"{'---' * 20}\n沒有交點 !")
            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 9:  # 不定積分/定積分/梯形法則
            print_result(f"{'---' * 20}\n不定積分 定積分 梯形法則近似值\n{'---' * 20}")

            Sum = sp.Symbol('x', real=True)

            try:
                integral_result = sp.integrate(func(Sum), Sum)
                print_result(f"不定積分結果：{integral_result}")
            except Exception as e:
                print_result(f"不定積分計算失敗：{e}")
            print_result("---" * 20)

            a = float(params['a'])
            b = float(params['b'])

            try:
                result_real = sp.integrate(func(Sum), (Sum, a, b)).evalf()
            except ValueError as e:
                print_result(f"❌ 定積分計算失敗：{e}")
                result_real = None

            Choice = params.get('Choice', 'n')
            if Choice == 'y':
                n = int(params['n_trap'])
                h = (b - a) / n
                print_result(f"{'---' * 20}\n公式 : 子區間長度 * (頭+尾 + 2 *(中間項數)) /2")
                print_result(f"{'---' * 20}\n每個子區間的長度Δx : {h}")
                first = func(a).evalf()
                last = func(b).evalf()
                print_result(f"頭:{first}")
                sum_ANS = 0

                if n > 1:
                    for i in range(1, n):
                        ANS = 2 * func(a + i * h).evalf()
                        print_result(str(ANS))
                        sum_ANS += ANS
                else:
                    print_result("區間數=1，無中間點")

                print_result(f"尾:{last}")
                trapezoid_result = (first + last + sum_ANS) * h / 2
                print_result(
                    f"{'---' * 20}\n梯形法則的近似结果: {trapezoid_result} (準確至四位小數:{trapezoid_result:.4f})")
                if result_real is not None:
                    print_result(f"定積分計算準確結果: {result_real} (準確至四位小數:{result_real:.4f})")
                    if result_real > trapezoid_result:
                        print_result("低估值")
                    elif abs(result_real - trapezoid_result) < 1e-9:
                        print_result("近似值與準確值相等")
                    else:
                        print_result("高估值")
            else:
                if result_real is not None:
                    print_result(f"{"---" * 20}\n定積分準確結果：{result_real} (準確至四位小數:{result_real:.4f})")

        # -------------------------------------------------------------------------------------------
        elif choice == 10:  # 1-4次函數因式分解/求根
            print_result(f"{'---' * 20}\n1-4次函數因式分解/求根 ( Ax^4 +Bx^3 + Cx^2 + Dx + E = 0 )\n{'---' * 20}")

            A = float(params.get('A_4', '0') or '0')
            B = float(params.get('B_3', '0') or '0')
            C = float(params.get('C_2', '0') or '0')
            D = float(params.get('D_1', '0') or '0')
            E = float(params.get('E_0', '0') or '0')

            A = sp.Rational(A).limit_denominator()
            B = sp.Rational(B).limit_denominator()
            C = sp.Rational(C).limit_denominator()
            D = sp.Rational(D).limit_denominator()
            E = sp.Rational(E).limit_denominator()

            Sum = sp.Symbol('x', real=True)
            fx = sp.simplify(A * Sum ** 4 + B * Sum ** 3 + C * Sum ** 2 + D * Sum + E)

            if A == 0 and B == 0 and C == 0:
                result_roots = sp.solve(fx, Sum, real=True)
                print_result(f"f(x) = 0 , x 的準確值 : {Fraction(float(result_roots[0])).limit_denominator()} "
                             f"(近似值(小數形式) ： {sp.N(result_roots[0])})")
            else:
                print_result(f"嘗試在實數範圍內因式分解 : {fx}")
                factor_result = sp.factor(fx)

                if factor_result == fx:
                    print_result(f"{'---' * 20}\n無法在有理數範圍內因式分解\n{'---' * 20}")
                else:
                    print_result(f"{'---' * 20}\n有理數範圍內因式分解結果 : {factor_result}\n{'---' * 20}")

                solve_result = sp.solve(fx, Sum, real=True)
                frac = []
                real_roots = []

                for r in solve_result:
                    simplified_r = sp.nsimplify(r)
                    try:
                        float(sp.N(simplified_r))
                        if simplified_r.is_real:
                            real_roots.append(simplified_r)
                    except (TypeError, ValueError):
                        continue

                if real_roots:
                    for i in real_roots:
                        if i.is_rational:
                            frac.append(i)
                            print_result(f"實數根 : {i} (是有理數)")
                        else:
                            frac.append(round(float(sp.N(i)), 9))
                            print_result(f"實數根 : {i}\n是無理數，近似值 = {round(float(sp.N(i)), 9)}")
                    print_result(f"f(x)=0 x 的準確值/近似值 : {frac}")
                else:
                    print_result("f(x)=0 沒有實數解")

            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 11:  # 進制轉換
            print_result(f"{'---' * 20}\n進制轉換 (2 , 10 , 16)\n{'---' * 20}")

            base = int(params['base'])
            num = params['num']

            if base == 2:
                decimal_num = int(num, 2)
                ANS_hex = hex(decimal_num)
                print_result('---' * 20)
                print_result(f"二進制 {num} → 十進制 {decimal_num}")
                print_result(f"二進制 {num} → 十六進制 {ANS_hex[2:].upper()}")
            elif base == 10:
                decimal_num = int(num)
                ANS_bin = bin(decimal_num)
                ANS_hex = hex(decimal_num)
                if decimal_num >= 0:
                    ANS_bin = ANS_bin[2:]
                    ANS_hex = ANS_hex[2:].upper()
                else:
                    ANS_bin = "-" + ANS_bin[3:]
                    ANS_hex = "-" + ANS_hex[3:].upper()
                print_result('---' * 20)
                print_result(f"十進制 {decimal_num} → 二進制 {ANS_bin}")
                print_result(f"十進制 {decimal_num} → 十六進制 {ANS_hex}")
            elif base == 16:
                hex_num = num.upper()
                ANS_decimal = int(hex_num, 16)
                ANS_bin = bin(ANS_decimal)
                if ANS_decimal >= 0:
                    ANS_bin = ANS_bin[2:]
                else:
                    ANS_bin = "-" + ANS_bin[3:]
                print_result('---' * 20)
                print_result(f"十六進制 {hex_num} → 二進制 {ANS_bin}")
                print_result(f"十六進制 {hex_num} → 十進制 {ANS_decimal}")
            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 12:  # 微分計算
            print_result(f"{'---' * 20}\n微分計算\n{'---' * 20}")

            Sum = sp.Symbol('x', real=True)
            fx = sp.simplify(custom_func(Sum))
            print_result(f"f(x) = {fx}")
            df1 = sp.diff(custom_func(Sum), Sum, 1)
            df2 = sp.diff(custom_func(Sum), Sum, 2)

            print_result(f"f'(x) = {sp.simplify(df1)}\nf''(x) = {sp.simplify(df2)}\n{'---' * 20}")
            real_solutions = []
            solutions_1 = sp.solve(df1, Sum, real=True)

            for r in solutions_1:
                simplified_r = sp.nsimplify(r, tolerance=1e-9)
                try:
                    float(sp.N(simplified_r))
                    if simplified_r.is_real and simplified_r.is_finite:
                        real_solutions.append(simplified_r)
                except (TypeError, ValueError, IndexError, sp.PolynomialDivisionFailed):
                    continue

            real_solutions = [i for i in real_solutions if abs(sp.N(df1.subs(Sum, i))) < 1e-9]
            real_solutions = sorted(list(set(real_solutions)))

            if len(real_solutions) == 0:
                print_result("f'(x) = 0 無實數解")
            elif len(real_solutions) >= 3:
                print_result(f"找到 {len(real_solutions)} 個臨界點（本程式最多處理2個）")
            else:
                print_result(f"f'(x) = 0  x 的值 :")
                if len(real_solutions) == 1:
                    try:
                        x_1 = sp.simplify(real_solutions[0])
                        print_result(f"x_1 = {x_1}\n{'---' * 20}")
                        left_x1 = sp.N(df1.subs(Sum, x_1 - 1e-9))
                        right_x2 = sp.N(df1.subs(Sum, x_1 + 1e-9))

                        a = "−" if left_x1 < -1e-9 else "+"
                        b = "−" if right_x2 < -1e-9 else "+"

                        table_data = [["  x  ", f"x<{x_1}", f"{x_1}", f"x>{x_1}"],
                                      ["f'(x)", a, "0", b]]

                        for row in table_data:
                            print_result("\t".join(map(str, row)))
                        print_result('---' * 20)

                        if a == "−" and b == "+":
                            print_result(
                                f"局部極小點：x = {x_1}\n極小值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 = {sp.N(custom_func(x_1)):.9f}")
                        elif a == "+" and b == "−":
                            print_result(
                                f"局部極大點：x = {x_1}\n極大值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 = {sp.N(custom_func(x_1)):.9f}")
                        else:
                            print_result(f"x = {x_1} 不是極值點（符號沒有變化）")
                    except Exception as e:
                        print_result(f"計算出錯 : 請自行計算 (實數解可能不是有效的) : {e}")
                elif len(real_solutions) == 2:
                    try:
                        x_1 = sp.simplify(real_solutions[0])
                        x_2 = sp.simplify(real_solutions[1])

                        print_result(f"x_1 = {x_1}\nx_2 = {x_2}\n{'---' * 20}")
                        left_x1 = sp.N(df1.subs(Sum, x_1 - 1e-9))
                        left_x2 = sp.N(df1.subs(Sum, x_2 - 1e-9))
                        right_x2 = sp.N(df1.subs(Sum, x_2 + 1e-9))

                        a = "-" if left_x1 < -1e-9 else "+"
                        b = "-" if left_x2 < -1e-9 else "+"
                        c = "-" if right_x2 < -1e-9 else "+"

                        table_data = [["x  ", f"x < x_1 ", f" x_1 ", f" x_1 < x < x_2 ", f" x_2 ", f"x > x_2 "],
                                      ["dy/dx", a, "0", b, "0", c]]

                        for row in table_data:
                            print_result("\t".join(map(str, row)))

                        print_result('---' * 20)
                        if a == "-" and b == "+" and c == "-":
                            print_result(
                                f"局部極小點：x = {x_1}\n極小值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 ≈ {sp.N(custom_func(x_1)):.9f}")
                            print_result('---' * 20)
                            print_result(
                                f"局部極大點：x = {x_2}\n極大值：f({sp.N(x_2)}) = {sp.simplify(custom_func(x_2))}\n近似值 ≈ {sp.N(custom_func(x_2)):.9f}")
                        elif a == "+" and b == "-" and c == "+":
                            print_result(
                                f"局部極大點：x = {x_1}\n極大值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 ≈ {sp.N(custom_func(x_1)):.9f}")
                            print_result('---' * 20)
                            print_result(
                                f"局部極小點：x = {x_2}\n極小值：f({sp.N(x_2)}) = {sp.simplify(custom_func(x_2))}\n近似值 ≈ {sp.N(custom_func(x_2)):.9f}")
                        elif a == "-" and b == "-" and c == "+":
                            print_result(f"x_1 不是極值點（符號沒有變化）")
                            print_result(
                                f"局部極小點：x = {x_2}\n極小值：f({sp.N(x_2)}) = {sp.simplify(custom_func(x_2))}\n近似值 ≈ {sp.N(custom_func(x_2)):.9f}")
                        elif a == "+" and b == "+" and c == "-":
                            print_result(f"x_1 不是極值點（符號沒有變化）")
                            print_result(
                                f"局部極大點：x = {x_2}\n極大值：f({sp.N(x_2)}) = {sp.simplify(custom_func(x_2))}\n近似值 ≈ {sp.N(custom_func(x_2)):.9f}")
                        elif a == "-" and b == "-" and c == "+":
                            print_result(f"x_2 不是極值點（符號沒有變化）")
                            print_result(
                                f"局部極小點：x = {x_1}\n極小值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 ≈ {sp.N(custom_func(x_1)):.9f}")
                        elif a == "+" and b == "-" and c == "-":
                            print_result(f"x_2 不是極值點（符號沒有變化）")
                            print_result(
                                f"局部極大點：x = {x_1}\n極大值：f({sp.N(x_1)}) = {sp.simplify(custom_func(x_1))}\n近似值 ≈ {sp.N(custom_func(x_1)):.9f}")
                        else:
                            print_result("兩點也不是極值")
                    except Exception as e:
                        print_result(f"計算出錯 : 請自行計算 (實數解可能不是有效的) : {e}")

            real_solutions_2 = []
            solutions_2 = sp.solve(df2, Sum, real=True)

            for r in solutions_2:
                simplified_r = sp.nsimplify(r)
                try:
                    if simplified_r.is_real and simplified_r.is_finite:
                        if abs(sp.N(df2.subs(Sum, simplified_r))) < 1e-9:
                            real_solutions_2.append(simplified_r)
                except Exception as e:
                    continue

            real_solutions_2 = [i for i in real_solutions_2 if abs(sp.N(df2.subs(Sum, i))) < 1e-9]
            real_solutions_2 = sorted(list(set(real_solutions_2)))

            for i in range(len(real_solutions_2)):
                real_solutions_2[i] = sp.simplify(real_solutions_2[i])
            if len(real_solutions_2) == 0:
                print_result("f''(x) = 0 無實數解")
            else:
                print_result(
                    f"f''(x) = 0 的所有實數解：{", ".join([f"{sol}（≈{sp.N(sol):.4f}）" for sol in real_solutions_2])}")

            print_result('---' * 20)

        # -------------------------------------------------------------------------------------------
        elif choice == 13:  # 函數化簡
            print_result(f"{'---' * 20}\n函數化簡/代入/合併同類項\n{'---' * 20}")

            Sum = sp.symbols("x")
            fx = custom_func(Sum)
            print_result(f"原方程 : {fx}")

            if fx == sp.expand(fx):
                print_result("方程已經可能是最簡 ( 可以嘗試因式分解，前往功能 [10] )!")
            else:
                print_result(f"展開/化簡後的方程 : {sp.expand(fx)}\n"
                             f"合併同類項後的方程 : {sp.collect(fx, Sum)}")
            print_result('---' * 20)

            n = params.get('n_sub', '')
            if n:
                try:
                    n = float(n)
                    ANS = custom_func(n)
                    print_result(
                        f"f({n}) = {ANS} (小數形式 : {sp.N(ANS)}) (分數形式 : {Fraction(float(ANS)).limit_denominator()})")
                    print_result('---' * 20)
                except Exception as e:
                    print_result(f"請輸入有效數字 / 該數值不在f(x)定義域內: {e}\n{'---' * 20}")

        # -------------------------------------------------------------------------------------------
        elif choice == 14:  # 統計學 E(x) Var(x)
            print_result(f"{'---' * 20}\n統計學 E(x) Var(x)\n{'---' * 20}")

            data = params.get('data', '').split(',')
            number_data = params.get('number_data', '').split(',')

            data = [float(d) for d in data if d.strip()]
            number_data = [float(nd) if nd.strip() else 1.0 for nd in number_data if nd.strip()]

            if len(number_data) != 0 and sum(number_data) != 0:
                Sum = 0
                for i in number_data:
                    Sum += i
                for i in range(len(number_data)):
                    number_data[i] = number_data[i] / Sum

                if 1e-9 < sum(number_data) < 1 + 1e-9:
                    mu = 0
                    mu_2 = 0

                    for i in range(len(data)):
                        mu += data[i] * number_data[i]
                        mu_2 += data[i] ** 2 * number_data[i]
                    Var = mu_2 - mu ** 2
                    Sigma = Var ** 0.5

                    print_result(f"次數統一化概率 : {number_data}\n"
                                 f"平均值 E(x) = {mu:.9f} (分數形式 : {Fraction(mu).limit_denominator()})\n"
                                 f"( E(x) ) ^2 = {mu ** 2 :.9f} (分數形式 : {Fraction(mu ** 2).limit_denominator()})\n"
                                 f"E(x^2) = {mu_2:.9f} (分數形式 : {Fraction(mu_2).limit_denominator()})\n"
                                 f"方差 Var(X) = {Var:.9f} (分數形式 : {Fraction(Var).limit_denominator()})\n"
                                 f"總體標準差 = {Sigma:.9f} (分數形式 : {Fraction(Sigma).limit_denominator()})")
                else:
                    print_result("所輸入的概率不等於 1 ，請重新輸入!")
            else:
                print_result("未得到數據，請重新輸入!")
            print_result('---' * 20)

        return {"status": "success", "result": "\n".join(result)}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================== Flask 路由 ===========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def api_calculate():
    choice = request.form.get('choice')
    params = request.form.to_dict()
    result = calculate(choice, params)
    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)