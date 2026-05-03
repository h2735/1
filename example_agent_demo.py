"""
example_agent_demo.py
示例代码：光学实验数据处理与AI辅助分析演示
Author: 林其磊
Date: 2025-10-22
Description:
    - 模拟光学实验数据（衍射图样）
    - 自动计算FFT / 相位
    - 可视化输出
    - 支持批量处理（多任务示例）
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_diffraction_pattern(N=512, wavelength=532e-9, slit_width=50e-6, screen_distance=1.0):
    x = np.linspace(-0.01, 0.01, N)
    k = 2 * np.pi / wavelength
    amplitude = np.sinc(k * slit_width * x / (2 * np.pi))
    intensity = np.abs(amplitude)**2
    return x, intensity

def analyze_fft(intensity):
    fft_result = np.fft.fftshift(np.fft.fft(intensity))
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude

def visualize_pattern(x, intensity, fft_magnitude):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(x*1e3, intensity)
    plt.title("Diffraction Pattern")
    plt.xlabel("x (mm)")
    plt.ylabel("Intensity (a.u.)")

    plt.subplot(1,2,2)
    plt.plot(x*1e3, fft_magnitude)
    plt.title("FFT of Intensity")
    plt.xlabel("Spatial Frequency (a.u.)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

def batch_process(num_samples=3):
    for i in range(num_samples):
        x, intensity = generate_diffraction_pattern()
        fft_magnitude = analyze_fft(intensity)
        print(f"Sample {i+1} processed.")
        visualize_pattern(x, intensity, fft_magnitude)

if __name__ == "__main__":
    batch_process(num_samples=3)
