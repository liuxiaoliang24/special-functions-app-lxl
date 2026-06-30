"""
特殊函数计算模块
实现勒让德多项式、连带勒让德函数、三类柱函数（第一类 J、第二类 Y、第三类 H）
基于递推关系或级数展开，所有函数支持向量化输入（x 为 numpy 数组）。
"""

import numpy as np
from math import factorial
from scipy.special import gamma  # 仅用于贝塞尔级数中的 Gamma 函数

# ======================== 勒让德多项式 ========================
def legendre(l, x):
    """
    计算勒让德多项式 P_l(x)
    参数：
        l : int, 非负整数阶数
        x : array_like, 自变量，定义域建议 [-1, 1]
    返回：
        y : ndarray，与 x 同形的 P_l(x) 值
    算法：递推公式
        P_0 = 1,  P_1 = x
        (l+1)P_{l+1}(x) = (2l+1)x P_l(x) - l P_{l-1}(x)
    """
    x = np.asarray(x, dtype=float)
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x.copy()
    
    # 初始化 P_0, P_1
    p_prev = np.ones_like(x)   # P_{n-1}, 起始为 P_0
    p_curr = x.copy()          # P_n, 起始为 P_1
    
    # 从 n=1 递推到 l-1，得到 P_l
    for n in range(1, l):
        # 递推公式：P_{n+1} = [(2n+1)x P_n - n P_{n-1}] / (n+1)
        p_next = ((2*n + 1) * x * p_curr - n * p_prev) / (n + 1)
        p_prev, p_curr = p_curr, p_next
    return p_curr


# ======================== 连带勒让德函数 ========================
def associated_legendre(l, m, x):
    """
    计算连带勒让德函数 P_l^m(x)
    参数：
        l : int, 非负整数，阶数
        m : int, 整数，次数，要求 |m| <= l，否则返回 0
        x : array_like
    返回：
        y : ndarray
    算法：
        - 若 m < 0，利用对称关系转换为正 m
        - 正 m 时，先计算起始值 P_m^m(x)，然后递推至 P_l^m(x)
        起始值：P_m^m(x) = (-1)^m (2m-1)!! (1 - x^2)^{m/2}
        递推（l >= m）：
            P_{m+1}^m = (2m+1) x P_m^m
            (l-m+1) P_{l+1}^m = (2l+1) x P_l^m - (l+m) P_{l-1}^m
    """
    l = int(l)
    m = int(m)
    x = np.asarray(x, dtype=float)
    
    # 定义域外或 m > l 时函数值为 0
    if abs(m) > l:
        return np.zeros_like(x)
    
    # 处理负 m 次：P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m
    if m < 0:
        factor = (-1)**abs(m) * factorial(l - abs(m)) / factorial(l + abs(m))
        return factor * associated_legendre(l, abs(m), x)
    
    # ----- 计算 P_m^m -----
    # 双阶乘 (2m-1)!! = 1·3·5·...·(2m-1)
    if m == 0:
        p_mm = np.ones_like(x)      # P_0^0 = 1
    else:
        double_fact = 1
        for i in range(1, m+1):
            double_fact *= (2*i - 1)
        p_mm = ((-1)**m) * double_fact * (1 - x**2)**(m/2.0)
    
    if l == m:
        return p_mm
    
    # ----- 计算 P_{m+1}^m -----
    p_mp1_m = (2*m + 1) * x * p_mm
    if l == m + 1:
        return p_mp1_m
    
    # ----- 递推到 l -----
    p_km2 = p_mm      # P_{k-1}^m 初始为 P_m^m
    p_km1 = p_mp1_m   # P_k^m 初始为 P_{m+1}^m
    # k 从 m+1 到 l-1，每次递推得到 P_{k+1}^m
    for k in range(m+1, l):
        # 公式：(k - m + 1) P_{k+1}^m = (2k+1) x P_k^m - (k+m) P_{k-1}^m
        p_kp1 = ((2*k + 1) * x * p_km1 - (k + m) * p_km2) / (k - m + 1)
        p_km2, p_km1 = p_km1, p_kp1
    return p_km1


# ======================== 贝塞尔函数（三类柱函数） ========================
def bessel_J(nu, x, max_terms=200):
    """
    第一类贝塞尔函数 J_ν(x)
    参数：
        nu : float, 阶数 (ν >= 0)
        x  : array_like, 自变量 (x > 0)
        max_terms : int, 级数最大项数（防止无限循环）
    返回：
        J : ndarray
    算法：级数展开
        J_ν(x) = Σ_{k=0}^{∞} (-1)^k / (k! Γ(ν+k+1)) * (x/2)^{ν+2k}
    使用递推计算每一项，当前项乘因子得到下一项，避免重复计算幂和阶乘。
    """
    nu = float(nu)
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)
    
    # 第一项 k=0
    term = (x/2.0)**nu / gamma(nu + 1.0)
    result = term.copy()
    
    # 递推求后续项： term_{k+1} = term_k * [ - (x/2)^2 / ((k+1)(ν+k+1)) ]
    for k in range(1, max_terms):
        term = term * (- (x/2.0)**2) / (k * (nu + k))
        result += term
        # 若所有点的增量绝对值已足够小，可提前终止
        if np.max(np.abs(term)) < 1e-15:
            break
    return result


def bessel_Y(nu, x, max_terms=200):
    """
    第二类贝塞尔函数（诺依曼函数）Y_ν(x)
    参数：
        nu : float, 阶数
        x  : array_like
        max_terms : int
    返回：
        Y : ndarray
    算法：
        对于非整数 ν，使用关系式
            Y_ν = (J_ν cos(νπ) - J_{-ν}) / sin(νπ)
        对于整数 ν，该式分母为 0，采用极限近似：用 ν' = ν + 1e-8 代替。
        此近似在工程范围内精度可接受，且避免了复杂的级数展开。
        注：也可通过递推精确计算整数阶，但本模块为演示目的，采用近似方案。
    """
    nu = float(nu)
    x = np.asarray(x, dtype=float)
    
    # 判断是否为整数（允许浮点误差）
    if abs(nu - round(nu)) < 1e-12:
        # 整数阶，用微小偏移量避免奇点
        eps = 1e-8
        nu_shifted = nu + eps
        J_nu = bessel_J(nu_shifted, x, max_terms)
        J_minus_nu = bessel_J(-nu_shifted, x, max_terms)
        Y = (J_nu * np.cos(nu_shifted * np.pi) - J_minus_nu) / np.sin(nu_shifted * np.pi)
    else:
        J_nu = bessel_J(nu, x, max_terms)
        J_minus_nu = bessel_J(-nu, x, max_terms)
        Y = (J_nu * np.cos(nu * np.pi) - J_minus_nu) / np.sin(nu * np.pi)
    return Y


def bessel_H(nu, x, kind=1, max_terms=200):
    """
    第三类贝塞尔函数（汉克尔函数）
        H_ν^{(1)}(x) = J_ν(x) + i Y_ν(x)
        H_ν^{(2)}(x) = J_ν(x) - i Y_ν(x)
    参数：
        nu : float
        x  : array_like
        kind : int, 1 或 2，指定第一类/第二类汉克尔函数
    返回：
        H : ndarray (复数)
    """
    J = bessel_J(nu, x, max_terms)
    Y = bessel_Y(nu, x, max_terms)
    if kind == 1:
        return J + 1j * Y
    elif kind == 2:
        return J - 1j * Y
    else:
        raise ValueError("kind 只能为 1 或 2")


def bessel_function(nu, x, bessel_type='J'):
    """
    统一调用接口
    参数：
        nu : float，阶数
        x  : array_like
        bessel_type : str, 取值为 'J', 'Y', 'H1', 'H2' 之一
    返回：
        函数值，'H1','H2' 时为复数数组
    """
    if bessel_type == 'J':
        return bessel_J(nu, x)
    elif bessel_type == 'Y':
        return bessel_Y(nu, x)
    elif bessel_type == 'H1':
        return bessel_H(nu, x, kind=1)
    elif bessel_type == 'H2':
        return bessel_H(nu, x, kind=2)
    else:
        raise ValueError("bessel_type 必须是 'J', 'Y', 'H1', 'H2' 之一")