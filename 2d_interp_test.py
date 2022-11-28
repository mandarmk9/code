#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

L = 2 * np.pi
Nx = 512
dx = L / Nx

Lt = 2
Nt = 300
x = np.arange(0, L, dx)
kx = np.fft.ifftshift(2.0 * np.pi / Lx * np.arange(-Nx/2, Nx/2))

dt = Lt / Nt
t = np.arange(0, Lt, dt)
kt = np.fft.ifftshift(2.0 * np.pi / Lt * np.arange(-Nt/2, Nt/2))

KX, KT = np.meshgrid(kx, kt)
X, T = np.meshgrid(x, t)
X[X < 0] += L
X[X > L/2] = - L + X[X > L/2]

X, T = np.meshgrid(x, t)
X[X < 0] += L
X[X > L/2] = - L + X[X > L/2]

T[T < 0] += L
T[T > L/2] = - L + T[T > L/2]


f = ((X)**4) * np.cos(T)
f_kx = np.fft.fft(f, axis=1)
df_dX = 12*((X**2)) * np.cos(T)
df_dX2 = np.abs(df_dX)**2
# print(f_kx.shape)
# print(f.shape)
# print(kx.shape)
# print(kx[None, :].shape)
# print(kx[:, None].shape)
# df_dX_num = np.zeros(shape=(f.shape), dtype=complex)
df_dX_num = np.fft.ifft(f_kx, axis=1) * ((1j * kx[None, :])**2)
df_dX_num2 = np.abs(df_dX_num)**2

fig, ax = plt.subplots()
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('t', fontsize=12)
plot2d_2 = ax.pcolormesh(x, t, df_dX2, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'${df}/{dX}$', fontsize=20)
ax.legend(loc='upper right')
legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
frame = legend.get_frame()
plt.tight_layout()
frame.set_edgecolor('white')
frame.set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')
plt.savefig('/vol/aibn31/data1/mandar/plots/schr_tests/df_dX.png')
plt.close()

fig, ax = plt.subplots()
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('t', fontsize=12)
plot2d_2 = ax.pcolormesh(x, t, df_dX_num2, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'$df/{dX}$', fontsize=20)
ax.legend(loc='upper right')
legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
frame = legend.get_frame()
plt.tight_layout()
frame.set_edgecolor('white')
frame.set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')
plt.savefig('/vol/aibn31/data1/mandar/plots/schr_tests/df_dX_num.png')
plt.close()
