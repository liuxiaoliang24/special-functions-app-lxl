"""
函数展开模块
支持：
- 勒让德级数展开：f(x) ≈ Σ c_l P_l(x)   (l=0..N)
- 连带勒让德级数展开：f(x) ≈ Σ c_l P_l^m(x) (l=|m|..N)
- 贝塞尔级数展开：f(x) ≈ Σ c_n J_ν(λ_n x / a) (n=1..N)
使用 sympy 解析用户输入的目标函数表达式，利用 scipy.integrate.quad 计算系数。
返回系数、近似函数、以及逐阶拟合 GIF。
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import jn_zeros  # 用于求贝塞尔函数零点
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import tempfile
import os

from special_funcs import legendre, associated_legendre, bessel_J

# ================== 字符串解析为数值函数 ==================
def parse_target_function(expr_str):
    """
    将用户输入的表达式字符串转换为可调用的数值函数 f(x)。
    使用 sympy 解析，支持常用数学函数和常量，如 sin, cos, exp, pi 等。
    返回：(f_callable, sympy_expr) 或 (None, error_msg)
    """
    try:
        x_sym = sp.symbols('x')
        # 将字符串转为 sympy 表达式，支持常用函数
        expr = sp.sympify(expr_str, locals={'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
                                            'pi': sp.pi, 'sqrt': sp.sqrt, 'log': sp.log,
                                            'x': x_sym})
        # 检查表达式中是否包含 x
        if not expr.has(x_sym):
            return None, "表达式未包含变量 x"
        # 转换为数值函数
        f = sp.lambdify(x_sym, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos,
                                                         'exp': np.exp, 'pi': np.pi,
                                                         'sqrt': np.sqrt, 'log': np.log}])
        return f, None
    except Exception as e:
        return None, f"表达式解析错误: {str(e)}"

# ================== 勒让德展开 ==================
def legendre_expand(f, N, domain=(-1, 1)):
    """
    计算函数 f 在 [-1,1] 上的勒让德级数展开系数 (c0..cN)。
    系数公式：c_l = (2l+1)/2 * ∫_{-1}^{1} f(x) P_l(x) dx
    返回：coeffs (list), 近似函数 approx_func(x)
    """
    coeffs = []
    for l in range(N + 1):
        integrand = lambda x, l=l: f(x) * legendre(l, x)
        integral, _ = quad(integrand, domain[0], domain[1], limit=200, epsabs=1e-10)
        c = (2*l + 1) / 2 * integral
        coeffs.append(c)
    
    def approx(x):
        s = np.zeros_like(x, dtype=float)
        for l, c in enumerate(coeffs):
            s += c * legendre(l, x)
        return s
    
    return coeffs, approx

# ================== 连带勒让德展开 ==================
def associated_legendre_expand(f, m, N, domain=(-1, 1)):
    """
    固定 m，计算连带勒让德级数展开，l 从 |m| 到 N。
    系数：c_l = (2l+1)/2 * (l-m)!/(l+m)! * ∫_{-1}^{1} f(x) P_l^m(x) dx
    返回：coeffs (list，索引从0对应l=|m|), approx_func
    """
    lmin = abs(m)
    if N < lmin:
        raise ValueError(f"最大阶数 N ({N}) 不能小于 |m| ({lmin})")
    coeffs = []
    for l in range(lmin, N + 1):
        integrand = lambda x, l=l: f(x) * associated_legendre(l, m, x)
        integral, _ = quad(integrand, domain[0], domain[1], limit=200, epsabs=1e-10)
        # 计算归一化系数
        from math import factorial
        norm = (2*l + 1) / 2 * factorial(l - abs(m)) / factorial(l + abs(m))
        c = norm * integral
        coeffs.append(c)
    
    def approx(x):
        s = np.zeros_like(x, dtype=float)
        for idx, l in enumerate(range(lmin, N + 1)):
            s += coeffs[idx] * associated_legendre(l, m, x)
        return s
    
    return coeffs, approx

# ================== 贝塞尔展开 ==================
def bessel_expand(f, nu, N, a, bessel_type='J'):
    """
    在区间 [0, a] 上，利用第一类贝塞尔函数 J_ν 的正交性展开。
    使用第 n 个正零点 λ_n （J_ν(λ_n)=0）。
    系数：c_n = 2/(a^2 * J_{ν+1}^2(λ_n)) * ∫_0^a x f(x) J_ν(λ_n x / a) dx
    参数：
        nu : 贝塞尔阶数（float）
        N  : 展开项数（正整数）
        a  : 区间右端点
    返回：coeffs (list), 零点列表 zeros, approx_func
    """
    # 求前 N 个正零点
    zeros = jn_zeros(nu, N)  # 返回 1D 数组
    coeffs = []
    for n in range(N):
        lam = zeros[n]
        # 计算积分 ∫_0^a x f(x) J_ν(λ_n x / a) dx
        integrand = lambda x, lam=lam: x * f(x) * bessel_J(nu, lam * x / a)
        integral, _ = quad(integrand, 0, a, limit=200, epsabs=1e-10)
        # 归一化因子
        J_next = bessel_J(nu + 1, lam)
        norm = 2 / (a**2 * J_next**2)
        c = norm * integral
        coeffs.append(c)
    
    def approx(x):
        s = np.zeros_like(x, dtype=float)
        for n in range(N):
            lam = zeros[n]
            s += coeffs[n] * bessel_J(nu, lam * x / a)
        return s
    
    return coeffs, zeros, approx

# ================== 生成展开的 LaTeX 表达式 ==================
def expansion_latex(func_type, coeffs, **kwargs):
    """
    将系数列表转换为 LaTeX 字符串展示。
    返回 str，例如：'0.5 P_0(x) - 0.3 P_2(x) + ...'
    """
    terms = []
    if func_type == 'legendre':
        for l, c in enumerate(coeffs):
            if abs(c) < 1e-10:
                continue
            sign = '+' if c >= 0 else '-'
            if len(terms) == 0 and sign == '+':
                sign = ''
            c_abs = abs(c)
            terms.append(f"{sign} {c_abs:.4f} P_{{{l}}}(x)")
    elif func_type == 'associated_legendre':
        m = kwargs.get('m', 0)
        lmin = abs(m)
        for idx, c in enumerate(coeffs):
            if abs(c) < 1e-10:
                continue
            l = lmin + idx
            sign = '+' if c >= 0 else '-'
            if len(terms) == 0 and sign == '+':
                sign = ''
            c_abs = abs(c)
            terms.append(f"{sign} {c_abs:.4f} P_{{{l}}}^{{{m}}}(x)")
    elif func_type == 'bessel':
        nu = kwargs.get('nu', 0)
        a = kwargs.get('a', 1)
        zeros = kwargs.get('zeros', [])
        for n, c in enumerate(coeffs):
            if abs(c) < 1e-10:
                continue
            sign = '+' if c >= 0 else '-'
            if len(terms) == 0 and sign == '+':
                sign = ''
            c_abs = abs(c)
            lam = zeros[n]
            terms.append(f"{sign} {c_abs:.4f} J_{{{nu}}}\\left(\\frac{{{lam:.4f} x}}{{{a}}}\\right)")
    else:
        return "无法识别的展开类型"
    if not terms:
        return "0"
    return ' '.join(terms)

# ================== 逐阶拟合 GIF ==================
def create_expansion_gif(func_type, f, max_order, domain, **kwargs):
    """
    生成动态展示逐阶逼近效果的 GIF。
    func_type : 'legendre', 'associated_legendre', 'bessel'
    f : 目标函数（可调用）
    max_order : 最大阶数（对勒让德类）或最大项数（贝塞尔）
    domain : (x_min, x_max) 绘图区间
    kwargs : 
        - 对勒让德：无
        - 对连带勒让德：m=...
        - 对贝塞尔：nu=..., a=...
    返回 BytesIO 对象
    """
    x = np.linspace(domain[0], domain[1], 500)
    f_vals = f(x)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 预计算 y 范围
    y_min = min(np.min(f_vals), 0)
    y_max = max(np.max(f_vals), 0)
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.plot(x, f_vals, 'k--', linewidth=1, alpha=0.7, label='目标函数 f(x)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # 根据类型确定阶数序列
    if func_type == 'legendre':
        orders = list(range(max_order + 1))
    elif func_type == 'associated_legendre':
        m = kwargs.get('m', 0)
        lmin = abs(m)
        if max_order < lmin:
            raise ValueError("最大阶数不能小于 |m|")
        orders = list(range(lmin, max_order + 1))
    elif func_type == 'bessel':
        orders = list(range(1, max_order + 1))  # 项数从1开始
    else:
        raise ValueError("未知函数类型")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
    approx_line, = ax.plot([], [], lw=2, color='red', label='当前近似')
    
    def animate(i):
        if func_type == 'legendre':
            current_order = orders[i]
            # 重新计算到 current_order 的系数和近似函数
            coeffs, approx = legendre_expand(f, current_order, domain)
        elif func_type == 'associated_legendre':
            current_order = orders[i]
            coeffs, approx = associated_legendre_expand(f, m, current_order, domain)
        else:  # bessel
            current_terms = orders[i]
            coeffs, zeros, approx = bessel_expand(f, kwargs['nu'], current_terms, kwargs['a'])
        
        y_approx = approx(x)
        approx_line.set_data(x, y_approx)
        ax.set_title(f'逐阶近似 (当前阶/项数: {orders[i]})')
        # 更新图例
        ax.legend([f'目标函数 f(x)', f'近似 (阶={current_order if func_type!="bessel" else current_terms})'])
        return approx_line,
    
    ani = FuncAnimation(fig, animate, frames=len(orders), interval=800, repeat=True)
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
        tmp_path = tmpfile.name
    ani.save(tmp_path, writer='pillow', fps=1.25)
    with open(tmp_path, 'rb') as f_gif:
        gif_bytes = io.BytesIO(f_gif.read())
    os.unlink(tmp_path)
    plt.close(fig)
    return gif_bytes