"""
工具模块：提供参数校验、阶数列表生成和默认配置。
"""

def validate_legendre_params(l, m=None):
    """
    校验勒让德/连带勒让德参数
    l : int, 非负整数
    m : int 或 None
    返回错误信息字符串，若无错误则返回 None
    """
    if not isinstance(l, int) or l < 0:
        return "阶数 l 必须为非负整数"
    if m is not None:
        if not isinstance(m, int):
            return "次数 m 必须为整数"
        if abs(m) > l:
            return f"|m| ({abs(m)}) 不能大于 l ({l})"
    return None

def validate_bessel_params(nu, bessel_type):
    """
    校验柱函数参数
    nu : float, 阶数，要求非负实数
    bessel_type : str, 'J','Y','H1','H2'
    返回错误信息字符串，若无错误则返回 None
    """
    if nu < 0:
        return "贝塞尔阶数 ν 必须为非负实数"
    if bessel_type not in ('J', 'Y', 'H1', 'H2'):
        return "柱函数类型必须是 J, Y, H1, H2 之一"
    return None

def parse_order_list(order_str, max_order=None):
    """
    将用户输入的阶数字符串解析为列表。
    支持格式：
        - 逗号分隔的数字，如 "0,1,2,5"
        - 单个数字，如 "3"
    若输入为空，则使用 max_order 生成 0,1,...,max_order 的列表。
    返回 (orders, error)
    """
    order_str = order_str.strip()
    if not order_str:
        # 无输入，尝试使用最大阶数
        if max_order is None:
            return [], "请输入阶数列表或设定最大阶数"
        # 生成整数序列 0,1,...,max_order
        max_order = int(max_order)
        return list(range(max_order + 1)), None
    
    # 尝试按逗号分割
    parts = order_str.split(',')
    orders = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            # 支持整数或小数（贝塞尔阶数可为小数）
            if '.' in part:
                orders.append(float(part))
            else:
                orders.append(int(part))
        except ValueError:
            return [], f"无法解析 '{part}' 为数字，请检查输入格式"
    
    if len(orders) == 0:
        return [], "阶数列表为空，请至少输入一个阶数"
    return orders, None

def get_default_x_range(func_type):
    """返回默认绘图区间"""
    if func_type in ('legendre', 'associated_legendre'):
        return -1.0, 1.0
    else:
        return 0.1, 20.0

def validate_x_range(x_min, x_max, func_type):
    """校验并返回合法的 x 轴范围"""
    x_min = float(x_min)
    x_max = float(x_max)
    if x_min >= x_max:
        raise ValueError("x_min 必须小于 x_max")
    if func_type in ('legendre', 'associated_legendre'):
        # 超出 [-1,1] 也可以计算，但提示警告
        pass
    else:
        if x_min <= 0:
            # 柱函数 x 应大于0，自动修正
            x_min = 0.01
    return x_min, x_max