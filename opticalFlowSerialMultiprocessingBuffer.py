import time 
import serial
import numpy as np
import cv2
from scipy import stats
import csv
import multiprocessing
from Queue import Queue

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


ser = serial.Serial('/dev/cu.usbserial-A7006RGw', 9600, timeout = 10) # Establish the connection on a specific port
cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps = " + str(fps))
capturas = []
manager = multiprocessing.Manager()
cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
#ns = manager.Namespace()



def avanzar(y, e):
	#print("mande instruccion al motor")
	inicio_movimiento = time.time()
	ser.write(str(y)) # Convert the decimal number to ASCII then send it to the Arduino
	ser.write('\n'.encode()) #mandar new line para avisar que termino de mandar la instruccion de avance
	getSerialValue = ser.readline() #
	tiempo_movimiento = time.time()-inicio_movimiento
	print("tiempo movimiento = " + str(tiempo_movimiento))
	#print("recibi confirmacion de vovimiento completo")
	e.set()	
	#print '\nValor retornado de Arduino: %s' % (getSerialValue)
 
def procesar_secuencia(secuencia):
	#print("procesando secuencia " + str(len(secuencia)))
	p0 = reset_features(secuencia[0])
	desplazamiento_acumulado = 0
	desplazamiento_entre_frames = 0
	for i in range(len(secuencia)-1):
		desplazamiento_entre_frames, p0, img = avance_entre_cuadros_consecutivos(secuencia[i], secuencia[i+1], p0)
		print("desplazamiento entre frames = " + str(desplazamiento_entre_frames))
		desplazamiento_acumulado = desplazamiento_acumulado + desplazamiento_entre_frames
		cv2.imshow('preview',img)
		cv2.waitKey(0)
	return desplazamiento_acumulado


def reset_features(primer_frame_secuencia):

	print("____RESETEANDO FEATURES______")
	# Tomar el primer frame de la secuencia y buscar features para trackear
	old_frame = primer_frame_secuencia
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	return p0

def avance_entre_cuadros_consecutivos(vieja, nueva, p0):
	old_gray = cv2.cvtColor(vieja, cv2.COLOR_BGR2GRAY)	#convierto a escala de grises
	new_gray = cv2.cvtColor(nueva, cv2.COLOR_BGR2GRAY)	#convierto a escala de grises
	mask = np.zeros_like(vieja)	#defino una mascara del tamano de la imagen para poder dibujar arriba los puntos

	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

	good_new = p1[st==1]
	good_old = p0[st==1]

	diferencia = []

	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		
		diferencia.append(int(c-a))
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		mask = cv2.circle(mask,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(nueva,mask)

	moda = stats.mode(diferencia)
	m = moda[0]
	p0 = good_new.reshape(-1,1,2)
	return m, p0, img


e = multiprocessing.Event()
e.clear()
p1 = multiprocessing.Process(target=avanzar, args=(0, e,))# crear un proceso paralelo para el avance del motor 	
p1.start() #disparar el avance del motor
p1.join()

while(True):

	inicio = time.time()
	ret,frame_big = cap.read()
	#global capturas
	capturas.append(frame_big)
	tiempo_while = time.time()-inicio
	print("tiempo while = " + str(tiempo_while))
	
	#entra en este if cuando se alcanza la posicion de destino (termina el movimiento)
	if(e.is_set()):
		e.clear() #borrar la variable que se usa para la comunicacion con el otro thread
		print("se capturaron " + str(len(capturas)) + " imagenes")
		avance_del_film = procesar_secuencia(capturas)
		print("desplazamiento total secuencia = " + str(avance_del_film))	
		#global capturas
		capturas = []
		#time.sleep(2)
		p1 = multiprocessing.Process(target=avanzar, args=(1, e,))# crear un proceso paralelo para el avance del motor 	
		p1.start() #disparar un nuevo avance del motor
	



	#p2.join() #esperar a que termine la captura

	
	


	#avanzar(10)
	#capturar()

	if cv2.waitKey(20) & 0xFF == 27:
		break



