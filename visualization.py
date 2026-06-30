"""
可视化模块
提供：
  - 单阶函数曲线
  - 多阶对比图
  - 阶数渐变动图（GIF）
  - 下载支持（PNG 和 GIF 字节流）
"""

import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import os
from matplotlib.animation import FuncAnimation

from special_funcs import legendre, associated_legendre, bessel_function

# 设置全局绘图参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 尝试支持中文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False   # 关闭外部 LaTeX，避免缺失字体导致解析错误

# ======================== 辅助：生成自变量和标签 ========================
def get_x_range(func_type, x_range=None):
    """根据函数类型返回合理的默认 x 区间和点数"""
    if x_range is None:
        if func_type in ('legendre', 'associated_legendre'):
            x_range = (-1, 1)
        else:
            x_range = (0.1, 20)  # 柱函数避免 x=0 奇异
    return np.linspace(x_range[0], x_range[1], 500)

def get_func_label(func_type, params):
    """生成用于图例的函数标签（使用 mathtext，避免外部 LaTeX）"""
    if func_type == 'legendre':
        l = params.get('l', 0)
        return f'$P_{{{l}}}(x)$'
    elif func_type == 'associated_legendre':
        l = params.get('l', 0)
        m = params.get('m', 0)
        return f'$P_{{{l}}}^{{{m}}}(x)$'
    elif func_type == 'bessel':
        nu = params.get('nu', 0)
        btype = params.get('bessel_type', 'J')
        nu_str = str(int(nu)) if isinstance(nu, float) and nu.is_integer() else str(nu)
        if btype == 'J':
            return f'$J_{{{nu_str}}}(x)$'
        elif btype == 'Y':
            return f'$Y_{{{nu_str}}}(x)$'
        elif btype == 'H1':
            return f'$H_{{{nu_str}}}^{{(1)}}(x)$'
        elif btype == 'H2':
            return f'$H_{{{nu_str}}}^{{(2)}}(x)$'
    return ''

def compute_function(func_type, l, m, nu, bessel_type, x):
    """
    根据类型和参数调用对应的特殊函数。
    注意：勒让德/连带勒让德使用 l, m；贝塞尔使用 nu, bessel_type。
    """
    if func_type == 'legendre':
        return legendre(l, x)
    elif func_type == 'associated_legendre':
        return associated_legendre(l, m, x)
    elif func_type == 'bessel':
        return bessel_function(nu, x, bessel_type)
    else:
        raise ValueError(f"未知函数类型: {func_type}")

# ======================== 单阶曲线 ========================
def plot_single(func_type, l=0, m=0, nu=0, bessel_type='J', x_range=None):
    """
    绘制单一阶数的函数曲线。
    返回 matplotlib Figure 对象，供 st.pyplot() 使用。
    """
    x = get_x_range(func_type, x_range)
    y = compute_function(func_type, l, m, nu, bessel_type, x)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    label = get_func_label(func_type, {'l': l, 'm': m, 'nu': nu, 'bessel_type': bessel_type})
    ax.plot(x, np.real(y), label=label, linewidth=2)  # 复数取实部
    
    # 如果柱函数选择了 H1/H2，且 y 为复数，可选绘制虚部
    if func_type == 'bessel' and bessel_type in ('H1', 'H2'):
        ax.plot(x, np.imag(y), '--', label=label + ' 虚部', alpha=0.7)
    
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('x')          # 去掉 $...$ 避免解析错误
    ax.set_ylabel('y')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ======================== 多阶对比图 ========================
def plot_multi(func_type, orders, m=0, bessel_type='J', x_range=None):
    """
    同一坐标系绘制多个阶数的曲线。
    参数 orders: 列表，元素为 l (勒让德类) 或 nu (贝塞尔类)。
    """
    x = get_x_range(func_type, x_range)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
    
    for i, order in enumerate(orders):
        if func_type in ('legendre', 'associated_legendre'):
            l = int(order)
            nu = 0
            lbl_params = {'l': l, 'm': m, 'nu': 0, 'bessel_type': None}
        else:  # bessel
            nu = float(order)
            l = 0
            lbl_params = {'l': 0, 'm': 0, 'nu': nu, 'bessel_type': bessel_type}
        
        y = compute_function(func_type, l, m, nu, bessel_type, x)
        label = get_func_label(func_type, lbl_params)
        ax.plot(x, np.real(y), color=colors[i], linewidth=2, label=label)
    
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Multi-order Comparison Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ======================== GIF 动图生成 ========================
def create_gif(func_type, max_order, m=0, bessel_type='J', x_range=None, 
               hold_history=True, interval=500):
    """
    生成阶数逐渐变高的 GIF 动图，并返回 GIF 的字节流。
    参数：
        max_order : int, 最大阶数，将绘制从 0 到 max_order 的每一帧。
        hold_history : bool, True 则保留之前阶的曲线（半透明），False 仅绘制当前阶。
        interval : 帧间隔（毫秒）
    返回：
        gif_bytes : BytesIO 对象，可直接用于 st.download_button 或 st.image
    """
    x = get_x_range(func_type, x_range)
    orders = list(range(max_order + 1))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 预计算 y 轴范围，避免帧间抖动
    all_y = []
    for order in orders:
        if func_type in ('legendre', 'associated_legendre'):
            y = compute_function(func_type, order, m, 0, None, x)
        else:
            y = compute_function('bessel', 0, 0, float(order), bessel_type, x)
        all_y.append(np.real(y))
    y_min = min(np.min(y) for y in all_y)
    y_max = max(np.max(y) for y in all_y)
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
    
    def animate(i):
        order = orders[i]
        ax.set_title(f'degree: {order}')
        if func_type in ('legendre', 'associated_legendre'):
            y = compute_function(func_type, order, m, 0, None, x)
            lbl = get_func_label(func_type, {'l': order, 'm': m, 'nu': 0, 'bessel_type': None})
        else:
            nu = float(order)
            y = compute_function('bessel', 0, 0, nu, bessel_type, x)
            lbl = get_func_label('bessel', {'l': 0, 'm': 0, 'nu': nu, 'bessel_type': bessel_type})
        
        if hold_history:
            for line in ax.lines[:]:
                line.remove()
            for j in range(i):
                prev_order = orders[j]
                if func_type in ('legendre', 'associated_legendre'):
                    prev_y = compute_function(func_type, prev_order, m, 0, None, x)
                else:
                    nu_prev = float(prev_order)
                    prev_y = compute_function('bessel', 0, 0, nu_prev, bessel_type, x)
                ax.plot(x, prev_y, color=colors[j], alpha=0.4, linewidth=1)
            ax.plot(x, np.real(y), color=colors[i], linewidth=2, label=lbl)
        else:
            ax.clear()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.plot(x, np.real(y), color=colors[i], linewidth=2, label=lbl)
            ax.legend()
        
        return ax.lines
    
    ani = FuncAnimation(fig, animate, frames=len(orders), interval=interval, repeat=True)
    
    # 使用临时文件保存 GIF，避免扩展名未知错误
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
        tmp_path = tmpfile.name
    ani.save(tmp_path, writer='pillow', fps=1000/interval)
    with open(tmp_path, 'rb') as f:
        gif_bytes = io.BytesIO(f.read())
    os.unlink(tmp_path)
    plt.close(fig)
    return gif_bytes

# ======================== 下载辅助函数 ========================
def fig_to_bytes(fig, format='png'):
    """
    将 matplotlib Figure 转换为字节流，便于下载。
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf