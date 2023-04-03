#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

def x3(x):
    return np.power(x, 3)

# －5から5まで100個のデータ
x = np.linspace(-5, 5, 100)
y = x3(x)
plt.plot(x, y, label="$x^3$")
plt.legend() # 凡例の表示

# In[2]:


def calc_change_rate(f, a, h):
    # 関数fと位置a、幅hを引数にとって、
    # 変化量の割合を返す
    return (f(a+h) - f(a))/h


# In[3]:


x = np.linspace(-5, 5, 100)
# －5から5までの整数（全部で11個）
a = np.arange(-5, 5+1)
r = calc_change_rate(f=x3, a=a, h=1)
plt.plot(x, x3(x), label="$x^3$")
plt.plot(a, r, 'bo', label="change rate")
plt.legend()


# In[5]:


r = calc_change_rate(f=x3, a=x, h=0.0001)
plt.plot(x, x3(x), label="$x^3$")
plt.plot(x, r, label="change rate")
plt.legend()


# In[6]:


calc_change_rate(x3, 0, 1)


# In[7]:


calc_change_rate(x3, 0, 0.001)


# In[8]:


calc_change_rate(x3, 0, 0.00001)


# In[9]:


calc_change_rate(x3, 0, 1.0e-400)


# In[10]:


x = np.linspace(0, 13, 100)
n = np.arange(1, 13)

def s(t):
    # 混乱がないように引数をtにします
    return 50*t*(t+1)

plt.plot(x, s(x), label="$S(x)$")
plt.plot(n, s(n), 'bo', label="$S(n)$")
plt.legend()


# In[11]:


x = np.linspace(-1.1, 1.1, 100)

def f(x):
    return x**3 - x

def diff_f(x):
    return 3 * x**2 - 1

def zero(x):
    '''y=0に水平な直線を引くための関数'''
    return np.zeros(x.shape[0])

plt.plot(x, f(x), label="$f(x)=x^3 - x$")
plt.plot(x, diff_f(x), label="$f^{\prime}(x)=3x^{2}- 1$")
plt.plot(x, zero(x), label="y=0")
plt.legend()


# In[15]:


from scipy import optimize

optimize.minimize_scalar(f, bounds=(-1, 1))
# SciPyのバージョンによっては、method引数を正しく指定する必要があります。
#optimize.minimize_scalar(f, bounds=(-1, 1), method='bounded')


# In[16]:


x = np.linspace(-5, 5, 100)
plt.plot(x, np.exp(x), label="$e^{x}$")
plt.legend()


# In[17]:


x = np.linspace(-5, 5, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.plot(x, sigmoid(x), label=r"$\frac{1}{1+e^{-x}}$")
plt.legend()


# In[18]:


from sympy import var, exp, diff, init_printing
# 実行環境に応じて数式を見やすく表示してくれる
init_printing()
# 変数xを定義します
var("x")
# シグモイド関数を作ります
f = 1 / (1 + exp(-x))
# 関数fをxで微分します
diff(f, x)


# In[19]:


x = np.linspace(-5, 5, 100)

def diff_sigmoid(x):
    '''シグモイド関数の微分'''
    return sigmoid(x)*(1 - sigmoid(x))

plt.plot(x, sigmoid(x), label=r"$\frac{1}{1+e^{-x}}$")
plt.plot(x, diff_sigmoid(x), label=r"$\frac{e^{-x}}{(1+e^{-x})^{2}}$")
plt.legend()


# In[21]:


from mpl_toolkits.mplot3d import Axes3D

x1 = np.linspace(-4, 4, 20)
x2 = np.linspace(-4, 4, 20)
x1, x2 = np.meshgrid(x1, x2)

def f(x1, x2):
    return (x1+x2)

z = f(x1, x2)
fig, ax = plt.subplots(figsize=(7,7),
subplot_kw={"projection": "3d"})
ax.plot_surface(x1, x2, z, cmap="plasma")
ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
plt.show()              


# In[22]:


def sigmoid2(x1, x2):
    return 1 / (1 + np.exp(-1 * (x1+x2)))

z = sigmoid2(x1, x2)
fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"projection": "3d"})
ax.plot_surface(x1, x2, z, cmap="plasma")
ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
plt.show()


# In[23]:


fig, ax = plt.subplots()
im = ax.contourf(x1, x2, z, cmap='plasma')
plt.colorbar(im)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


# In[32]:


var('x1 x2')
f = 1 / (1 + exp(-x1 - x2))
diff(f, x1)


# In[33]:


diff(f, x2)


# In[24]:


x1 = np.linspace(-4, 4, 10)
x2 = np.linspace(-4, 4, 10)
x1, x2 = np.meshgrid(x1, x2)
z = sigmoid2(x1, x2)

def diff_sigmoid2(x1, x2):
    return np.exp(-1 * (x1+x2))/((1+np.exp(-1*(x1+x2)))**2)

u = diff_sigmoid2(x1, x2)
v = diff_sigmoid2(x1, x2)
fig, ax = plt.subplots()
im = ax.contourf(x1, x2, z, cmap='plasma')
plt.colorbar(im)
# 矢印を書くためのコード
plt.quiver(x1, x2, u, v)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


# In[25]:


x1 = np.linspace(-2, 2, 20)
x2 = np.linspace(-2, 2, 20)
x1, x2 = np.meshgrid(x1, x2)

def cone(x1, x2):
    return 1 - np.exp(-(x1**2 + x2**2))

z = cone(x1, x2)
fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"projection": "3d"})
ax.plot_surface(x1, x2, z, cmap="plasma")
ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
plt.show()


# In[26]:


var('x1 x2')
f = 1 - exp(-x1**2 - x2**2)
diff(f, x1)


# In[27]:


diff(f, x2)


# In[28]:


x1 = np.linspace(-2, 2, 10)
x2 = np.linspace(-2, 2, 10)
x1, x2 = np.meshgrid(x1, x2)
z = cone(x1, x2)

def diff_cone(x1, x2, on='x1'):
    '''どちらの変数で微分しているかで結果を変更'''
    if on == 'x1':
        return 2 * x1 * np.exp(-1 * (x1**2 + x2** 2))
    return 2 * x2 * np.exp(-1 * (x1**2 + x2** 2))

u = diff_cone(x1, x2)
v = diff_cone(x1, x2, on='x2')
fig, ax = plt.subplots()
im = ax.contourf(x1, x2, z, cmap='plasma')
plt.colorbar(im)
plt.quiver(x1, x2, u, v)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


# In[ ]:




