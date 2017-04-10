import numpy as np
import cv2
from scipy import stats

cap = cv2.VideoCapture('output.avi')
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


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

desplazamiento = 0
m = 0
m_previa = 0
m_previa_previa = 0

font = cv2.FONT_HERSHEY_SIMPLEX

def acumular_desplazamiento():
	print('MOVIENDOSE')
	global desplazamiento
	desplazamiento = desplazamiento + m



def detenido():
	global desplazamiento
	global frame
	cv2.putText(frame,str(desplazamiento),(10,300), font, 1,(0,255,0),2)

	print('desplazamiento del ultimo avance = ' + str(desplazamiento))
	desplazamiento = 0
	

def reset_features():

	print("____RESETEANDO FEATURES______")
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	
	global p0 
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	global desplazamiento
	desplazamiento = 0


while(cap.isOpened()):

	#old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)
	
	#tomar el cuadro siguiente para intentar buscar los mismos corners
	#generar una copia en escala de grises
	
	#print("moda = " + str(m))
	#print("moda previa = " + str(m_previa))

	#detectar cuando el film termina de avanzar y se frena. En este momento se tomarian las imagenes para procesar. 
	#tambien en este momento se redefinen good features to track

	if (m == 0 and m_previa == 0 and m_previa_previa != 0):
		reset_features()

	ret,frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# calcular optical flow usando las dos imagenes en escala de grises (la anterior = old_gray y la actual = frame_gray)
	# la posicion de los puntos en la imagen actual se guarda en p1	
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]
	
	print("----------------------")

	# draw the tracks
	diferencia = []
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		#print(int(c-a))
		diferencia.append(int(c-a))
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		mask = cv2.circle(mask,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(frame,mask)
		if desplazamiento == 0: 
			cv2.putText(frame,"FRENADO",(10,300), font, 1,(0,0,255),2)	
		else:
			cv2.putText(frame,"MOVIENDOSE " + str(desplazamiento[0]),(10,300), font, 1,(0,255,0),2)	
		
		cv2.imshow('frame',img)	
    	if cv2.waitKey(20) & 0xFF == 27:
    		break

	#buscar la moda de las distancia entre los puntos trackeados
	m_previa_previa = m_previa
	m_previa = m
	moda = stats.mode(diferencia)
	m = moda[0]

	
	if m != 0: 
		acumular_desplazamiento()
	elif (m == 0 and m_previa == 0 and m_previa_previa == 0):
		detenido()


	old_frame = frame
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	#p0 = p1
	    # Now update the previous frame and previous points
	    #old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	
	#raw_input('Press enter to continue: ')

cv2.destroyAllWindows()
cap.release()