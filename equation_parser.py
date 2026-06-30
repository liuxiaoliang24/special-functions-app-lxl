"""
方程与解析模块
功能：
    1. 根据函数类型和参数，生成对应微分方程及通解的 LaTeX 字符串
    2. 解析用户输入的简写表达式（如 P_3(x), P_3^1(x), J_2.5(x)），
       提取函数类型和阶数/次数
"""

import re

# ---------- 方程与通解模板 ----------
def get_equation_and_solution(func_type, **params):
    """
    返回 (equation_latex, solution_latex)
    params 根据 func_type 包含相应键：
        - legendre : l
        - associated_legendre : l, m
        - bessel : nu, bessel_type
    """
    if func_type == 'legendre':
        l = params.get('l', 0)
        eq = r"(1-x^2)y'' - 2xy' + %d(%d+1)y = 0" % (l, l)
        sol = r"y = A\,P_{%d}(x) + B\,Q_{%d}(x)" % (l, l)
    
    elif func_type == 'associated_legendre':
        l = params.get('l', 0)
        m = params.get('m', 0)
        eq = r"(1-x^2)y'' - 2xy' + \left[%d(%d+1) - \frac{%d}{1-x^2}\right]y = 0" % (l, l, m**2)
        sol = r"y = A\,P_{%d}^{%d}(x) + B\,Q_{%d}^{%d}(x)" % (l, m, l, m)
    
    elif func_type == 'bessel':
        nu = params.get('nu', 0)
        # 将 nu 格式化为漂亮字符串（整数则不显示小数点）
        nu_str = str(int(nu)) if isinstance(nu, float) and nu.is_integer() else str(nu)
        eq = r"x^2 y'' + x y' + (x^2 - %s^2)y = 0" % nu_str
        btype = params.get('bessel_type', 'J')
        if btype in ('J', 'Y'):
            sol = r"y = A\,J_{%s}(x) + B\,Y_{%s}(x)" % (nu_str, nu_str)
        elif btype in ('H1', 'H2'):
            sol = r"y = A\,H_{%s}^{(1)}(x) + B\,H_{%s}^{(2)}(x)" % (nu_str, nu_str)
        else:
            sol = r"y = A\,J_{%s}(x) + B\,Y_{%s}(x)" % (nu_str, nu_str)
    
    else:
        raise ValueError("未知函数类型: " + func_type)
    
    return eq, sol


# ---------- 简写表达式解析 ----------
def parse_expression(expr):
    """
    解析如 P_3(x), P_3^1(x), J_2.5(x), Y_0(x), H1_1(x), H2_0.5(x) 的字符串
    返回字典：
        {
            'func_type': 'legendre' / 'associated_legendre' / 'bessel',
            'l': int or None,        # 勒让德阶数
            'm': int or None,        # 连带勒让德次数
            'nu': float or None,     # 贝塞尔阶数
            'bessel_type': 'J'/'Y'/'H1'/'H2'
        }
    解析失败返回 None
    """
    expr = expr.replace(' ', '')   # 去除空格
    # 匹配模式
    # 连带勒让德：P_l^m(x)  字母P，下划线l，上标^m，(x)
    pattern_legendre_assoc = re.compile(
        r'^P_(\d+)(?:\^(\d+))?\(x\)$'
    )
    # 柱函数：J_nu(x) / Y_nu(x) / H1_nu(x) / H2_nu(x)
    pattern_bessel = re.compile(
        r'^([JY]|H[12])_(\d+(?:\.\d+)?)\(x\)$'
    )
    
    # 先尝试匹配勒让德/连带勒让德
    m = pattern_legendre_assoc.match(expr)
    if m:
        l = int(m.group(1))
        sup = m.group(2)
        if sup is not None:
            # 连带勒让德
            m_val = int(sup)
            return {
                'func_type': 'associated_legendre',
                'l': l,
                'm': m_val,
                'nu': None,
                'bessel_type': None
            }
        else:
            # 纯勒让德
            return {
                'func_type': 'legendre',
                'l': l,
                'm': 0,
                'nu': None,
                'bessel_type': None
            }
    
    # 再尝试匹配柱函数
    m = pattern_bessel.match(expr)
    if m:
        prefix = m.group(1)   # 'J', 'Y', 'H1', 'H2'
        nu_str = m.group(2)
        nu = float(nu_str)
        bessel_type = prefix
        return {
            'func_type': 'bessel',
            'l': None,
            'm': None,
            'nu': nu,
            'bessel_type': bessel_type
        }
    
    # 均不匹配
    return None