"""
交互应用模块 —— 主程序
使用 Streamlit 搭建界面，集成勒让德函数、连带勒让德函数、三类柱函数的计算与可视化，
同时支持将任意函数展开为这些正交基的级数（勒让德、连带勒让德、贝塞尔展开）。
运行方式：在终端执行  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- 导入自定义模块 ----------
from special_funcs import legendre, associated_legendre, bessel_function
from equation_parser import get_equation_and_solution, parse_expression
from visualization import (
    plot_single, plot_multi, create_gif, fig_to_bytes, get_func_label
)
from utils import (
    validate_legendre_params, validate_bessel_params,
    parse_order_list, get_default_x_range, validate_x_range
)
from expansion import (
    parse_target_function,
    legendre_expand, associated_legendre_expand, bessel_expand,
    expansion_latex, create_expansion_gif
)

# ---------- 页面设置 ----------
st.set_page_config(page_title="特殊函数交互演示", layout="wide")
st.title("特殊函数交互式可视化与展开")
st.markdown("""
勒让德函数、连带勒让德函数、三类柱函数（贝塞尔函数族）的计算、图形展示及函数展开。
支持手动输入参数或使用 **简写表达式**（如 `P_3(x)`, `J_2.5(x)`）快速设定函数，
也可在“函数展开”标签页中将任意函数展开为正交基的级数。
""")

# ======================== 侧边栏：函数可视化参数 ========================
st.sidebar.header("函数选择与参数")
func_type = st.sidebar.selectbox(
    "选择函数类型",
    ["勒让德 (P_l)", "连带勒让德 (P_l^m)", "柱函数 (J, Y, H)"]
)

func_map = {
    "勒让德 (P_l)": "legendre",
    "连带勒让德 (P_l^m)": "associated_legendre",
    "柱函数 (J, Y, H)": "bessel"
}
func_key = func_map[func_type]

# 参数初始化
l_val = 0
m_val = 0
nu_val = 0.0
bessel_type_val = 'J'

if func_key in ("legendre", "associated_legendre"):
    st.sidebar.subheader("勒让德/连带勒让德参数")
    l_val = st.sidebar.number_input("阶数 l", min_value=0, value=2, step=1)
    if func_key == "associated_legendre":
        m_val = st.sidebar.number_input("次数 m", min_value=0, value=1, step=1)
    else:
        m_val = 0
else:
    st.sidebar.subheader("柱函数参数")
    nu_val = st.sidebar.number_input("阶数 ν", min_value=0.0, value=0.0, step=0.5)
    bessel_type_val = st.sidebar.selectbox("柱函数类型", ["J", "Y", "H1", "H2"])

# ---------- 侧边栏：简写表达式输入 ----------
st.sidebar.markdown("---")
st.sidebar.markdown("或直接输入简写表达式（将覆盖上面参数）：")
expr_input = st.sidebar.text_input("例如: P_3(x), P_3^1(x), J_2.5(x), Y_0(x)", value="")

if expr_input.strip():
    parsed = parse_expression(expr_input)
    if parsed is None:
        st.sidebar.error("表达式格式错误，请参考示例")
    else:
        func_key = parsed['func_type']
        if func_key == 'legendre':
            l_val = parsed['l']
            m_val = 0
            nu_val = 0.0
            bessel_type_val = 'J'
        elif func_key == 'associated_legendre':
            l_val = parsed['l']
            m_val = parsed['m']
            nu_val = 0.0
            bessel_type_val = 'J'
        elif func_key == 'bessel':
            nu_val = parsed['nu']
            bessel_type_val = parsed['bessel_type']
            l_val = 0
            m_val = 0
        st.sidebar.success(f"已识别：{expr_input}")

# ---------- 参数校验 ----------
param_error = None
if func_key in ('legendre', 'associated_legendre'):
    param_error = validate_legendre_params(l_val, m_val if func_key == 'associated_legendre' else None)
else:
    param_error = validate_bessel_params(nu_val, bessel_type_val)

params_valid = param_error is None
if not params_valid:
    st.sidebar.error(param_error)

# ---------- 侧边栏：绘图设置 ----------
st.sidebar.markdown("---")
st.sidebar.header("绘图设置")
default_x = get_default_x_range(func_key)
x_min = st.sidebar.number_input("x 最小值", value=default_x[0])
x_max = st.sidebar.number_input("x 最大值", value=default_x[1])
try:
    x_min, x_max = validate_x_range(x_min, x_max, func_key)
except ValueError as e:
    st.sidebar.error(str(e))
    params_valid = False

st.sidebar.subheader("多阶与动画阶数设定")
max_order = st.sidebar.number_input("最大阶数（用于生成序列 0 ~ max_order）", min_value=0, value=5, step=1)
order_list_str = st.sidebar.text_input("或手动输入阶数列表（逗号分隔，如 0,1,2,5）", value="")
orders, order_error = parse_order_list(order_list_str, max_order if not order_list_str else None)
if order_error:
    st.sidebar.error(order_error)

hold_history = st.sidebar.checkbox("GIF 保留历史曲线（半透明）", value=True)
interval = st.sidebar.slider("GIF 帧间隔 (ms)", 100, 1000, 500)

# ======================== 主区域：两个标签页 ========================
tab1, tab2 = st.tabs(["📈 函数可视化", "📊 函数展开"])

# --------------------- 标签页1：原有可视化功能 ---------------------
with tab1:
    if not params_valid:
        st.warning("请先在侧边栏修正参数错误，以显示图形。")
    else:
        # ---- 方程与通解 ----
        st.header("微分方程与通解")
        params = {}
        if func_key == 'legendre':
            params = {'l': l_val}
        elif func_key == 'associated_legendre':
            params = {'l': l_val, 'm': m_val}
        else:
            params = {'nu': nu_val, 'bessel_type': bessel_type_val}
        eq, sol = get_equation_and_solution(func_key, **params)
        st.latex(eq)
        st.latex(sol)

        # ---- 单阶函数图 ----
        st.header("单阶函数图像")
        try:
            fig_single = plot_single(
                func_key, l=l_val, m=m_val, nu=nu_val, bessel_type=bessel_type_val,
                x_range=(x_min, x_max)
            )
            st.pyplot(fig_single)
            # 下载按钮
            col1, col2 = st.columns(2)
            with col1:
                png_bytes = fig_to_bytes(fig_single, 'png')
                label = get_func_label(func_key, {'l': l_val, 'm': m_val, 'nu': nu_val, 'bessel_type': bessel_type_val})
                st.download_button(
                    label="下载 PNG",
                    data=png_bytes,
                    file_name=f"{func_key}.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"绘制单阶图时出错：{e}。请检查参数。")
            st.stop()

        # ---- 多阶对比图 ----
        st.header("多阶对比图")
        if orders and len(orders) > 0:
            try:
                fig_multi = plot_multi(
                    func_key, orders, m=m_val, bessel_type=bessel_type_val,
                    x_range=(x_min, x_max)
                )
                st.pyplot(fig_multi)
                st.download_button(
                    label="下载多阶对比图 (PNG)",
                    data=fig_to_bytes(fig_multi),
                    file_name=f"multi_{func_key}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"绘制多阶图时出错：{e}")
        else:
            st.warning("请在侧边栏设定阶数列表后生成多阶对比图")

        # ---- GIF 动图 ----
        st.header("阶数渐变动图")
        if orders and len(orders) > 1:
            try:
                with st.spinner("正在生成 GIF 动图，请稍候..."):
                    gif_bytes = create_gif(
                        func_key, max_order=max_order,
                        m=m_val, bessel_type=bessel_type_val,
                        x_range=(x_min, x_max),
                        hold_history=hold_history,
                        interval=interval
                    )
                st.image(gif_bytes, caption="阶数逐渐升高时的函数变化")
                st.download_button(
                    label="下载 GIF 动图",
                    data=gif_bytes,
                    file_name=f"animation_{func_key}.gif",
                    mime="image/gif"
                )
            except Exception as e:
                st.error(f"生成 GIF 时出错：{e}")
        else:
            st.info("请确保最大阶数 ≥ 1 或多阶列表包含至少两个元素，以生成动图")

        st.markdown("---")
        st.markdown("""
        **说明**：勒让德函数和连带勒让德函数定义在 $[-1,1]$；贝塞尔函数 $x>0$，汉克尔函数为复数值，绘图默认取实部。
        """)

# --------------------- 标签页2：函数展开功能 ---------------------
with tab2:
    st.header("正交基函数展开")
    st.markdown("""
    将任意目标函数 $f(x)$ 展开为所选正交基的级数形式，并观察逐阶逼近效果。
    支持勒让德级数、连带勒让德级数（固定 $m$）和贝塞尔级数（第一类 $J_\\nu$）。
    """)

    exp_type = st.selectbox(
        "选择基函数类型",
        ["勒让德 (P_l)", "连带勒让德 (P_l^m)", "贝塞尔 (J_ν)"]
    )
    exp_map = {
        "勒让德 (P_l)": "legendre",
        "连带勒让德 (P_l^m)": "associated_legendre",
        "贝塞尔 (J_ν)": "bessel"
    }
    exp_key = exp_map[exp_type]

    func_str = st.text_input(
        "输入目标函数 f(x)（例如 sin(pi*x), exp(-x**2), x^2 + 1）",
        value="sin(pi*x)"
    )
    f, parse_err = parse_target_function(func_str)
    if parse_err:
        st.error(parse_err)
        st.stop()

    col1, col2, col3 = st.columns(3)
    if exp_key == "legendre":
        N = col1.number_input("最大阶数 N", min_value=0, value=5)
        a_val = None; nu_val = None; m_exp = None
        domain_exp = (-1, 1)
    elif exp_key == "associated_legendre":
        m_exp = col1.number_input("固定次数 m", min_value=0, value=1)
        N = col2.number_input("最大阶数 N (≥|m|)", min_value=abs(m_exp), value=5)
        a_val = None; nu_val = None
        domain_exp = (-1, 1)
    else:
        nu_val = col1.number_input("阶数 ν", min_value=0.0, value=0.0, step=0.5)
        N = col2.number_input("展开项数 N", min_value=1, value=5)
        a_val = col3.number_input("区间右端点 a", min_value=0.1, value=1.0)
        m_exp = None
        domain_exp = (0, a_val)

    if st.button("计算展开"):
        with st.spinner("正在计算展开系数..."):
            try:
                if exp_key == "legendre":
                    coeffs, approx = legendre_expand(f, N, domain_exp)
                    zeros_list = None
                elif exp_key == "associated_legendre":
                    coeffs, approx = associated_legendre_expand(f, m_exp, N, domain_exp)
                    zeros_list = None
                else:
                    coeffs, zeros_list, approx = bessel_expand(f, nu_val, N, a_val)
            except Exception as e:
                st.error(f"展开计算失败：{e}")
                st.stop()

        # 系数表格
        st.subheader("展开系数")
        import pandas as pd
        coeff_data = {"阶/项": [], "系数": []}
        if exp_key == "legendre":
            for l, c in enumerate(coeffs):
                coeff_data["阶/项"].append(l)
                coeff_data["系数"].append(c)
        elif exp_key == "associated_legendre":
            lmin = abs(m_exp)
            for idx, c in enumerate(coeffs):
                coeff_data["阶/项"].append(lmin + idx)
                coeff_data["系数"].append(c)
        else:
            for n, c in enumerate(coeffs):
                coeff_data["阶/项"].append(n + 1)
                coeff_data["系数"].append(c)
        st.dataframe(pd.DataFrame(coeff_data))

        # LaTeX 表达式
        st.subheader("近似表达式")
        latex_str = expansion_latex(
            exp_key, coeffs,
            m=m_exp, nu=nu_val, a=a_val, zeros=zeros_list
        )
        st.latex(r"f(x) \approx " + latex_str)

        # 对比图
        st.subheader("原函数与近似对比")
        x_plot = np.linspace(domain_exp[0], domain_exp[1], 500)
        y_true = f(x_plot)
        y_approx = approx(x_plot)
        fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
        ax_comp.plot(x_plot, y_true, 'k--', linewidth=1.5, label='目标函数 f(x)')
        ax_comp.plot(x_plot, y_approx, 'r-', linewidth=2, label='近似函数')
        ax_comp.set_xlabel('x')
        ax_comp.set_ylabel('y')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        st.pyplot(fig_comp)

        # 逐阶逼近动画
        st.subheader("逐阶逼近动画")
        gif_kwargs = {}
        if exp_key == "associated_legendre":
            gif_kwargs['m'] = m_exp
        elif exp_key == "bessel":
            gif_kwargs['nu'] = nu_val
            gif_kwargs['a'] = a_val
        try:
            with st.spinner("正在生成逐阶逼近动画..."):
                gif_exp = create_expansion_gif(exp_key, f, N, domain_exp, **gif_kwargs)
            st.image(gif_exp, caption="阶数/项数逐渐增加时的拟合变化")
            st.download_button(
                label="下载逼近动画 GIF",
                data=gif_exp,
                file_name="expansion.gif",
                mime="image/gif"
            )
        except Exception as e:
            st.error(f"生成展开动画失败：{e}")