import time
import numpy as np
import random

# from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import sys
sys.path.append("../PyOssRoom_master/")

# Import Systems
import struct
import io
import os
import sys
import math
import platform
import pickle
# Import Audio
import pyaudio
import librosa
import soundfile

import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 픽셀 수
HEIGHT = 7  # 그리드월드 세로
WIDTH = 7  # 그리드월드 가로


def calculate_RT60(slope_VAL=0.15, c_param=None, data_in=None, tgt_rt60=None, a_param=None, fs=None):

	p_0dB = c_param.s_0dB
	p_10dB = c_param.s_10dB
	p_20dB = c_param.s_20dB
	p_30dB = c_param.s_30dB

	if a_param.RT60[0][0] > tgt_rt60:
		print(" Initial RT >= Target RT !!!")
		print("      ", a_param.RT60[0][0])

		# case 5
		gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
		gain_slope_b = np.logspace( 0, -slope_VAL, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
		gain_slope_c = np.logspace( -slope_VAL, -slope_VAL-0.1, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB
		# gain_slope_c = np.logspace( 0, -slope_VAL-0.01, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB      (0.1->0.01)
		# gain_slope_c = np.logspace( 0, -slope_VAL-0.001, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB   (0.1->0.001)
		gain_slope_d = np.ones((data_in.shape[0]-p_30dB), dtype='f') # (Reverberation)
		gain_slope = np.append( gain_slope_a, gain_slope_b)
		gain_slope = np.append( gain_slope, gain_slope_c)
		gain_slope = np.append( gain_slope, gain_slope_d)

	else:
		# case 5
		gain_slope_a = np.ones(p_0dB, dtype='f') # 시작점(0dB)까지
		gain_slope_b = np.logspace( 0, slope_VAL, num=( p_10dB-p_0dB ) ) # 0dB ~ -10dB(EDT)
		gain_slope_c = np.logspace( slope_VAL, slope_VAL+0.1, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB
		# gain_slope_c = np.logspace( 0, slope_VAL+0.01, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB   (0.1->0.01)
		# gain_slope_c = np.logspace( 0, slope_VAL+0.001, num=( p_30dB-p_10dB ) )# -10dB ~ -30dB   (0.1->0.001)
		gain_slope_d = np.ones((data_in.shape[0]-p_30dB), dtype='f') # (Reverberation)
		gain_slope = np.append( gain_slope_a, gain_slope_b)
		gain_slope = np.append( gain_slope, gain_slope_c)
		gain_slope = np.append( gain_slope, gain_slope_d)

	data_temp = data_in * gain_slope
	data_out, decay_out, a_param, c_param  = \
							learn.learning_decay(data_temp, fs)

	# dbg.dPlotAudio(fs, gain_slope, y_range=1.6, title_txt='gain slope', label_txt='gain slope', \
	# 		xl_txt='Time(sec)', yl_txt='Amplitude' )
	# dbg.dSavePlotAudio(fs, gain_slope, y_range=1.6, title_txt='gain slope', label_txt='gain slope', \
	# 		xl_txt='Time(sec)', yl_txt='Amplitude', newWindow=True, directory='./save_graph')

	return c_param, data_out, a_param, decay_out, gain_slope

class Env(tk.Tk):

	def __init__(self):

		super(Env, self).__init__()
		self.action_space = ['u', 'd', 'l', 'r']
		self.n_actions = len(self.action_space)

		#self.canvasWin = tk.Tk()

		self.title('Deep SARSA')
		self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
		self.shapes = self.load_images()
		self.canvas = self._build_canvas()

		self.counter = 0
		self.rewards = []
		self.goal = []

		self.texts = []

	def _build_canvas(self):

		canvas = tk.Canvas(self, bg='white',
						   height=HEIGHT * UNIT,
						   width=WIDTH * UNIT)
		# 그리드 생성
		for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
			x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
			canvas.create_line(x0, y0, x1, y1)
		for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
			x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
			canvas.create_line(x0, y0, x1, y1)

		self.rewards = []
		self.goal = []

	# 캔버스에 이미지 추가
		self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])  #(0,0)
		#self.circle = canvas.create_image(650, 650, image=self.shapes[1]) # 3*3
		#self.circle = canvas.create_image(450, 450, image=self.shapes[2]) # 5*5
		#self.circle = canvas.create_image(650, 650, image=self.shapes[2]) # 7*7
		canvas.pack()
		return canvas

	def _destroy_canvas(self):
		self.canvas.delete("string")
		#self.canvas.delete("all")
		return True

	def load_images(self):
		rectangle = PhotoImage(
			Image.open("../img/rectangle.png").resize((65, 65)))
		triangle = PhotoImage(
			Image.open("../img/triangle.png").resize((65, 65)))
		circle = PhotoImage(
			Image.open("../img/circle.png").resize((65, 65)))

		return rectangle, triangle, circle

	def text_value(self, row, col, contents, action, font='Helvetica', size=10,
				   style='normal', anchor="nw"):

		if action == 0:                      # u
			origin_x, origin_y = 7, 42
		elif action == 1:
			origin_x, origin_y = 84, 42      # d
		elif action == 2:
			origin_x, origin_y = 42, 5       # l
		else:
			origin_x, origin_y = 42, 67      # r

		x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
		font = (font, str(size), style)
		text = self.canvas.create_text(x, y, fill="black", text=contents,
									   font=font, anchor=anchor)
		return self.texts.append(text)

	def print_value_all(self, q_table):
		for i in self.texts:
			self.canvas.delete(i)
		self.texts.clear()
		for i in range(HEIGHT):
			for j in range(WIDTH):
				for action in range(0, 4):
					state = [i, j]
					if str(state) in q_table.keys():
						temp = q_table[str(state)][action]
						self.text_value(j, i, round(temp, 2), action)

	def coords_to_state(self, coords):
		x = int((coords[0] - 50) / 100)
		y = int((coords[1] - 50) / 100)
		return [x, y]

	def state_to_coords(self, state):
		x = int(state[0] * 100 + 50)
		y = int(state[1] * 100 + 50)
		return [x, y]

	def reset(self):
		self.update()
		# time.sleep(1)
		x, y = self.canvas.coords(self.rectangle)
		self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

		self.render()

		return self.coords_to_state(self.canvas.coords(self.rectangle))

	def step(self, action, c_param=None, data=None, tgt_rt60=None, a_param=None, fs=None):
		state = self.canvas.coords(self.rectangle)
		base_action = np.array([0, 0])
		self.render()

		reward = -5
		done = False

		std_RT = tgt_rt60

		decay = np.zeros(0)
		gain_slope = np.zeros(0)

		'''
		# RT60 계산
		#
		#
		'''

		if action == 0:  # u, 상, +0.5
			if state[1] > UNIT:
				base_action[1] -= UNIT
				slope_VAL = 0.01        #1.1
				c_param, data, a_param, decay, gain_slope = \
					calculate_RT60(slope_VAL=slope_VAL,
									c_param=c_param, data_in=data,
									tgt_rt60=tgt_rt60,
									a_param=a_param,
									fs=fs)
		# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				RT60 = a_param.RT60[0][0]
				# if RT60 >= std_RT:
				if (RT60 >= std_RT) and ( ( (RT60-std_RT)/std_RT ) < 0.001 ) :  # 시간오차 0.1% 이내
					done = True
				else:
					reward = +1
					# circle 위치 변경
			else:
				reward = -5

		elif action == 1:  # d, 하, -0.5
			if state[1] < (HEIGHT - 1) * UNIT:
				base_action[1] += UNIT
				slope_VAL = 0.02      #1.2
				c_param, data, a_param, decay, gain_slope = \
					calculate_RT60(slope_VAL=slope_VAL,
									c_param=c_param, data_in=data,
									tgt_rt60=tgt_rt60,
									a_param=a_param,
									fs=fs)

				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				RT60 = a_param.RT60[0][0]
				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				# if RT60 >= std_RT:
				if (RT60 >= std_RT) and ( ( (RT60-std_RT)/std_RT ) < 0.001 ) :  # 시간오차 0.1% 이내
					done = True
				else:
					reward = +2
			else:
				reward = -5

		elif action == 2:  # l, 좌, -0.3
			if state[0] > UNIT:
				base_action[0] -= UNIT
				slope_VAL = 0.03         #1.3
				c_param, data, a_param, decay, gain_slope = \
					calculate_RT60(slope_VAL=slope_VAL,
									c_param=c_param, data_in=data,
									tgt_rt60=tgt_rt60,
									a_param=a_param,
									fs=fs)

				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				RT60 = a_param.RT60[0][0]
				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				# if RT60 >= std_RT:
				if (RT60 >= std_RT) and ( ( (RT60-std_RT)/std_RT ) < 0.001 ) :  # 시간오차 0.1% 이내
					done = True
				else:
					reward = +5
			else:
				reward = -5

		elif action == 3:  # r, 우, +0.3
			if state[0] < (WIDTH - 1) * UNIT:
				base_action[0] += UNIT
				slope_VAL = 0.04      #1.4
				c_param, data, a_param, decay, gain_slope = \
					calculate_RT60(slope_VAL=slope_VAL,
									c_param=c_param, data_in=data,
									tgt_rt60=tgt_rt60,
									a_param=a_param,
									fs=fs)

				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				RT60 = a_param.RT60[0][0]
				# if RT60 = 1.6 이면 reward = 1000, circle위치 변경
				# if RT60 >= std_RT:
				if (RT60 >= std_RT) and ( ( (RT60-std_RT)/std_RT ) < 0.001 ) :  # 시간오차 0.1% 이내
					done = True
				else:
					reward = +10

			else:
				reward = -5
		else:
			reward = -5
			print("Action Error!!!")

		#print(action, base_action, state)
		#print("dBData = ", dBData)
		# 에이전트 이동
		self.canvas.move(self.rectangle, base_action[0], base_action[1])
		# 에이전트(빨간 네모)를 가장 상위로 배치
		self.canvas.tag_raise(self.rectangle)
		next_state = self.canvas.coords(self.rectangle)

		if done:
			reward = 10
			self.canvas.create_text(next_state[0], next_state[1], text="G", font="Times 20 bold underline", fill="blue", tag='string')
			print("... Done. RT60 = ", RT60)

		next_state = self.coords_to_state(next_state)
		if done:
			print("... Done. next_state = ", next_state)
		#print("decay : ",decay)
		return next_state, reward, done, c_param, data, a_param, decay, gain_slope

	def render(self):
		# time.sleep(0.03)
		self.update()
