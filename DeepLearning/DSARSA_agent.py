# 2020.11.26
# 음향 튜닝 파라미터 RT60 control
# 갈화학습 Q-learning 이용


import copy
import pylab
import sys
import random
import time
import datetime
from time import strftime
sys.path.append("../PyOssRoom_master/")
from DSARSA_environment import Env
from collections import defaultdict

# Import Systems
import struct
import io
import os
import sys
import math
import platform
import pickle
import pandas as pd

# Import Audio
import pyaudio
import librosa
import soundfile

import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt

import pandas as pd

# User Libraries
import pyOssWavfile
import pyRoomAcoustic as room
import pyOssDebug as dbg
import pyOssFilter
import pyOssLearn as learn

# Deep learning model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000

'''
1. _ori wave파일 만들기
2. Deep SARSA
3. _trans wav파일 만들기
'''

class DeepSARSAgent:
	def __init__(self, actions):

		self.load_model = False      # False = 1min 2sec, True = 44sec
		#self.load_model = False       # False = 1min 2sec, True = 44sec
		#self.load_model = True
		# 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
		# 상태의 크기와 행동의 크기 정의
		self.actions = actions
		self.action_size = len(self.actions)
		self.state_size = 2      # 15
		self.discount_factor = 0.99
		self.learning_rate = 0.01       # 0.001

		self.epsilon = 1.               # exploration
		self.epsilon_decay = .9999
		self.epsilon_min = 0.01
		self.model = self.build_model()

		if self.load_model:
			self.epsilon = 0.05
			# self.model.load_weights('./save_model/dsarsa_trained.h5')  deep_sarsa
			self.model.load_weights('./save_model/deep_sarsa.h5')

	# 상태가 입력 큐함수가 출력인 인공신경망 생성
	def build_model(self):
		model = Sequential()
		model.add(Dense(30, input_dim=self.state_size, activation='relu'))
		model.add(Dense(30, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	# 입실론 탐욕 방법으로 행동 선택
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			# 무작위 행동 반환
			return random.randrange(self.action_size)
		else:
			# 모델로부터 행동 산출
			state = np.reshape(state, [1, 2])

			state = np.float32(state)
			q_values = self.model.predict(state)
			return np.argmax(q_values[0])

	def train_model(self, state, action, reward, next_state, next_action, done):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		state = np.float32(state)
		next_state = np.float32(next_state)

		target = self.model.predict(state)[0]

		# print ("3-1. target = ", target)

		# 살사의 큐함수 업데이트 식
		if done:
			target[action] = reward
		else:
			target[action] = (reward + self.discount_factor *
							  self.model.predict(next_state)[0][next_action])

		# 출력 값 reshape
		target = np.reshape(target, [1, 4])
		# 인공신경망 업데이트
		self.model.fit(state, target, epochs=1, verbose=0)

	def load_model_weights(self, load_state):
		#if self.load_state:
		self.epsilon = 0.05
		self.model.load_weights('./save_model/deep_sarsa.h5')

def menu_rt60():
	print('\n\n**************************************************')
	print('RT값 입력'.center(38))
	print('**************************************************\n\n')

	input_RT = float(input(" 추가할 RT값 입력(sec.) : "))
	return input_RT


if __name__ == "__main__":

	input_RT = menu_rt60()
	if input_RT is None:
		exit()

	#str_fileinfo = '_mono_f32_44.1k'  # 파일명에 부가된 정보
	str_fileinfo = ""

	STAT_FILTER = False         # Filter Process Off: False, On: True

	STAT_SAVE_RESULT = True     # 결과물 저장 여부 선택 No Save: False, Save: True

	aud_name = "singing"               #1차
	#aud_name = "flute_music"
	#aud_name = "adult_female_speech"

	aud_dir     = 'audiofiles'
	imp_dir     = 'impulsefiles'
	result_dir  = 'resultfiles'

	excel_list = []

	env = Env()
	agent = DeepSARSAgent(actions=list(range(env.n_actions)))

	global g_current_file
	g_current_file = ""

	# Set Sampling Rate Frequency
	fs = 48000      # 48kHz   

	# Load audio
	aud_fname = aud_name + str_fileinfo         # str_fileinfo = ""
	audio_fname = pyOssWavfile.str_fname(aud_dir, aud_fname)  # 전체경로, 파일명 조합
	fmt_aud, data_aud, st_fmt_aud, t_aud = pyOssWavfile.readf32(audio_fname, samplerate=fs)

	# Process 
	for filename in os.listdir(imp_dir):
		if filename.endswith(".wav"):
			file_directory = os.path.join(imp_dir, filename)

			print("\n\n\n")
			print("file_directory = ", file_directory)

			g_current_file = filename       #global

			#pd.read_csv(file_directory)
			'''
			############################
			# 1. ori_ wave 파일 만들기
			# 예) singing_StairwayUniversityOfYork_mono_f32_44.1k_ori.wav
			############################

			# data_filt, decaycurve, a_param, c_param, st_fmt_w, imp_fname, fname_imp, init_RT60, data_a, \
			# aud_name, imp_name, str_fileinfo = make_original()
			'''

			#imp_name = "mh3_000_ortf_48k"
			imp_name = filename[:-4]

			#imp_fname = imp_name + str_fileinfo
			imp_fname = imp_name

			impulse_fname = pyOssWavfile.str_fname(imp_dir, imp_fname)  # 전체경로, 파일명 조합
			fmt_imp, data_imp, st_fmt_imp, t_imp = pyOssWavfile.readf32(impulse_fname, samplerate=fs)
			print("t_imp = ", t_imp)    # time, length

			# 3초짜리 데이터로 만듬(rt60=1.6 기준으로 함으로 3초이하 샘플을 3로로 늘려줌)
			# if t_imp < 3.0:
			#     t_temp = 3.0 - t_imp
			#     data = pyOssWavfile.insertSilence(data_imp, fs, t_temp)
			#     #print(data_w.shape[0] / st_fmt_w.fs)
			#     print("3초이하 데이터... ", data.shape[0]/fs)

			'''
			# 불러온 임펄스 파일의 음향 파라미터 특성 출력
			decay_imp = room.decayCurve(data_imp, estimate=None, fs=fs)
			#dbg.dPrintAParam(room.calcAcousticParam(data_imp, room.decayCurve(data_imp, estimate=None, fs=fs), fs))
			#dbg.dPrintAParam(room.calcAcousticParam(data_imp, room.decayCurve(data_imp, estimate=1.42, fs=fs), fs))
			'''

			###############################################################################
			# Filter Process to loaded impulse data and save filtered impulse
			###############################################################################

			# fc = 500  # Center freq for bandpass filter 500Hz

			# data_filt, decay, a_param, c_param = learn.learning_decay(False, data_imp, fs, fc, fname=impulse_fname)
			data_tmp, decay, a_param, c_param = learn.learning_decay(data_imp, fs)
			print("RT60 현재 값(초): ", a_param.RT60)

			# 원래 audio 파일 그리기 & Parameter
			#dbg.dPlotAudio(st_fmt_w.fs, data_filt, imp_fname + ' filtered - ORIGINAL ' + str(fc) + 'Hz', label_txt='',
			#           xl_txt="Time(sec)", yl_txt="Amplitude")
			#dbg.dPrintf(a_param.__dict__)
			#dbg.dPrintf(c_param.__dict__)

			# Save filtering impulse data
			# if STAT_SAVE_RESULT == True:
			#     imp_filt_fname = imp_name + '.filtered_' + str(fc) + 'Hz'
			#     sname_imp_filt = pyOssWavfile.str_fname(result_dir, imp_filt_fname)
			#     # dbg.dPrintf(sname_imp_filt)  # for debug
			#     pyOssWavfile.write(sname_imp_filt, fs, data_tmp)

			# Convolution Process with Anechoic audio data and Impulse or Filtered impulse data
			# if STAT_FILTER == True:
			#     data_convolve_ori = sig.fftconvolve(data_aud, data_filt)
			#     ori_name = imp_filt_fname
			# else:
			#     data_convolve_ori = sig.fftconvolve(data_aud, data_imp)
			#     ori_name = imp_fname
			# data_convolve_ori = sig.fftconvolve(data_aud, data_imp, mode="valid")
			data_convolve_ori = sig.oaconvolve(data_aud, data_imp, mode="full")
			ori_name = imp_fname

			# 비교
			init_RT60 = a_param.RT60

			if t_imp < init_RT60 + input_RT:
				t_temp = init_RT60 + input_RT - t_imp 
				# t_temp = input_RT 
				data_imp = pyOssWavfile.insertSilence(data_imp, fs, t_temp)
				#print(data_w.shape[0] / st_fmt_w.fs)
				print("input_RT보다 임펄스가 작을 때 변경된 임펄스 크기... ", data_imp.shape[0]/fs)


			# Save Original Wav File
			if STAT_SAVE_RESULT == True:
				sname_ori = pyOssWavfile.str_fname(result_dir, aud_name + '.ori.' + ori_name + '.RT60=' + str(init_RT60) ) # 파일경로 + 파일이름
				pyOssWavfile.write(sname_ori, fs, pyOssWavfile.normalize(data_convolve_ori))    # 무향실 음원에 필터링 된 임펄스를 적용한 wav file 저장
				print('* Save complete convolution data original')


			'''
			############################
			# 2. Deep SARSA
			############################
			'''

			# 필터링 된 임펄스를 사용할 것인지, 원래 임펄스를 사용할 것인지 결정에 따라 처리
			# if STAT_FILTER == True:
			#     init_data_filt = data_filt      # 강화학습에 사용할 임펄스 데이터는 '필터 처리 한 임펄스 데이터'
			#     trans_name = imp_filt_fname     # 강화학습 처리 한 음장처리 결과 파일 저장에 사용할 이름
			# else:
			#     init_data_filt = data_imp       # 사용할 임펄스 데이터가 원본 임펄스 데이터
			#     trans_name = imp_fname          # 강화학습 처리 한 음장처리 결과 파일 저장에 사용할 이름
			init_data = data_imp       # 사용할 임펄스 데이터가 원본 임펄스 데이터
			trans_name = imp_fname          # 강화학습 처리 한 음장처리 결과 파일 저장에 사용할 이름

			#init_data_filt = data_filt
			init_decay = decay
			init_a_param = a_param
			init_c_param = c_param
			#init_a_param_RT60 = init_RT60
			print(".......... RT60 init_a_param: ", init_a_param.RT60)
			print(".......... EDT  init_a_param: ", init_a_param.EDT)
			print(".......... D50  init_a_param: ", init_a_param.D50)
			print(".......... C50  init_a_param: ", init_a_param.C50)
			print(".......... C80  init_a_param: ", init_a_param.C80)

			#print('..... at first: init_a_param_RT60 = ', init_a_param_RT60)

			#env = Env()
			#agent = DeepSARSAgent(actions=list(range(env.n_actions)))

			global_step = 0
			scores, episodes = [], []

			start_time = time.time()
			str_start_time = strftime("%y-%m-%d %H:%M:%S")

			# 목표 RT 결정
			tgt_rt60 = init_RT60 + input_RT

			#for e in range(EPISODES):    # 1000
			for e in range(1):    # 1000, 1
				data_learn	= init_data
				decay		= init_decay
				a_param		= init_a_param
				c_param		= init_c_param

				done = False
				score = 0
				state = env.reset()
				# state = np.reshape(state, [1, 15])
				# print("episode: ", e, "====================")
				#print("First state = ", state)

				load_state = True
				#agent.load_model_weights(load_state)

				while True:
					# env.render()

					# env 초기화
					global_step += 1
					# 현재 상태에 대한 행동 선택
					state = np.reshape(state, [1, 2])
					action = agent.get_action(state)

					# 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
					# next_state, reward, done, c_param, data_learn, a_param, decay_learn = \
					# 		env.step(action, c_param=c_param, data=data_learn,
					# 				tgt_rt60=tgt_rt60, a_param=a_param,
					# 				fs=fs)

					next_state, reward, done, c_param, data_learn, a_param, decay_learn, gain_slope = \
							env.step(action, c_param=c_param, data=data_learn,
									tgt_rt60=tgt_rt60, a_param=a_param,
									fs=fs)

					next_state = np.reshape(next_state, [1, 2])
					next_action = agent.get_action(next_state)

					# 샘플로 모델 학습
					agent.train_model(state, action, reward, next_state, next_action, done)
					state = next_state
					score += reward
					state = copy.deepcopy(next_state)

					# Q-table 화면에 표시
					# env.print_value_all(agent.q_table)
					a_param_RT60 = a_param.RT60

					# if (a_param_RT60 >= tgt_rt60) or done:
					if ( (a_param_RT60 >= tgt_rt60) and ( (a_param_RT60-tgt_rt60)/tgt_rt60 < 0.001 ) ) or done:
						# 에피소드마다 학습 결과 출력
						scores.append(score)
						episodes.append(e)
						pylab.plot(episodes, scores, 'b')
						pylab.savefig("./save_graph/dsarsa_graph.png")
						#print("episode:", e, "  score:", score, "global_step",
						#      global_step, "  epsilon:", agent.epsilon)

						agent.model.save_weights("./save_model/deep_sarsa.h5")

						break


				# 100 에피소드마다 모델 저장
				if e % 100 == 0:
					agent.model.save_weights("./save_model/deep_sarsa.h5")

			print("End.")

			end_time = time.time()
			str_end_time = strftime("%y-%m-%d %H:%M:%S")
			print("")
			doing_sec = end_time - start_time
			doing_times  = str(datetime.timedelta(seconds = doing_sec))
			print("소요시간(H:M:S): ", doing_times, "\n\n")

			env._destroy_canvas()

			'''
			############################
			# 3. trans_ wave 파일 만들기
			# 예) singing_trans_StairwayUniversityOfYork_mono_f32_44.1k.wav
			############################

			# make_translate(st_fmt_w_fs, data_filt, imp_fname, init_RT60, data_a, \
							aud_name, imp_name, str_fileinfo)
			'''

			print("초기 RT60(초) : ", init_RT60)
			print("학습 후 RT60(초) : ", a_param_RT60, "\n\n")


			# dbg.dPlotAudio(fs, gain_slope)
			# dbg.dPlotAudio(fs, data_learn, title_txt=trans_name, label_txt=str(a_param_RT60), xl_txt='Time(sec)', yl_txt='Amplitude' )
			# dbg.dPlotDecay(fs, decay, ' decay curve of ' + trans_name, label_txt=str(a_param_RT60), xl_txt='Time(sec)', yl_txt='Amplitude' )
			dbg.dSavePlotAudio( fs, \
								data_learn, \
								title_txt = trans_name + '_waveform_' + str(init_RT60) + '_' + str(input_RT), \
								label_txt = 'RT60=' + str(a_param_RT60), \
								xl_txt = 'Time(sec)', \
								yl_txt = 'Amplitude', \
								newWindow = True, \
								directory = './'+result_dir )
			dbg.dSavePlotDecay( fs, \
								decay_learn, \
								title_txt = trans_name + '_Decay_' + str(init_RT60) + '_' + str(input_RT), \
								label_txt = 'RT60='+str(a_param_RT60), \
								xl_txt = 'Time(sec)', \
								yl_txt = 'Amplitude(dB)', \
								newWindow = True, \
								directory = './'+result_dir )
			dbg.dSavePlotAudio( fs, 
								librosa.amplitude_to_db(gain_slope), \
								y_range_min = -3.0, \
								y_range_max = 3.0, \
								title_txt = trans_name+'_slope_' + str(init_RT60) + '_' + str(input_RT), \
								label_txt = 'RT60=' + str(a_param_RT60), \
								xl_txt = 'Time(sec)', \
								yl_txt = 'Amplitude(dB)', \
								newWindow = True, \
								directory = './'+result_dir)
		
			# Convolution
			# data_convolve_learned = sig.fftconvolve(data_aud, data_learn, mode="valid")
			data_convolve_learned = sig.oaconvolve(data_aud, data_learn, mode="full")

			# Save Learned Impulse
			if STAT_SAVE_RESULT == True:
				imp_learn_fname = imp_name + '-' + str(init_RT60) + '-' + str(input_RT) + '-' + str(a_param_RT60)
				sname_imp_learn = pyOssWavfile.str_fname(result_dir, imp_learn_fname)
				pyOssWavfile.write( sname_imp_learn, fs, np.float32( pyOssWavfile.normalize(data_learn) ) )
				print('* Save complete learned impulse data')

			# Save Learning Processed Wav File
			if STAT_SAVE_RESULT == True:
				sname_trans = pyOssWavfile.str_fname(result_dir, \
								aud_name + '.trans.' + trans_name + '-' + str(init_RT60) + '-' + str(input_RT) + '-' + str(a_param_RT60))
				pyOssWavfile.write( sname_trans, fs, np.float32( pyOssWavfile.normalize(data_convolve_learned) ) )
				print('* Save complete convolution data trans')
			print('**************************************************')
			print('                Process Finished.')
			print('**************************************************', "\n\n")


			# 불러온 임펄스 파일의 음향 파라미터 특성 출력
			trans_a_param = a_param
			excel_list.append([imp_name, input_RT, init_a_param.RT60,  trans_a_param.RT60,
							init_a_param.EDT,  trans_a_param.EDT,
							init_a_param.D50, trans_a_param.D50,
							init_a_param.C50, trans_a_param.C50,
							init_a_param.C80, trans_a_param.C80])

		# 전체 끝난 후 결과 저장
		df = pd.DataFrame(excel_list, columns=['File_nm', 'Input RT(sec)', 'RT60_init', 'RT60_trans',
												'EDT_init', 'EDT_trans',
												'D50_init', 'D50_trans',
												'C50_init', 'C50_trans',
												'C80_init', 'C80_trans'])
		df.to_excel('excel_output.xlsx')
