import cv2
import numpy as np
import pickle
import sys

primera_ejecucion = int(sys.argv[1])
capturas = sys.argv[2]
contador_frame = sys.argv[3]

#original_inv = cv2.imread('nueva_prueba_captura.png')
original_inv = capturas[-1]

#invierto la imagen por como esta armado el dispositivo de captura
fimg=original_inv.copy()
fimg=cv2.flip(original_inv,0)
#fimg = cv2.resize(fimg, (0,0), fx=0.5, fy=0.5) #la reduzco para facilitar visualizacion
original = fimg. copy()

img = original.copy()
ROI = original.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
th1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	

#valores iniciales de las variables que definen la ROI y los extremos del film
centro_ROI = 40
alto_ROI = 10
treshold_borde = 127
borde_inferior = 0
borde_superior = 0
height, width, channels = original.shape
plot_height = 200

#dimensiones del film 8mm regular
frame_height = 4.88
frame_width = 3.68
inter_frame = 0.5

film_height = 7.975

hole_height = 1.829
hole_width = 1.270

hole_position = 0.902
inter_hole = 3.810-hole_width


frame_x = 0


square_wave = [0] * width
intensidad = [0] * width

d = 1 #paso para derivada primera


def definir_ROI(y):
	actualizar()

def ubicar_centro_ROI(y):
	actualizar()

def definir_treshold_borde(y):
	actualizar()

def actualizar():

	img = original.copy()
	ROI = original[centro_ROI-alto_ROI:centro_ROI+alto_ROI,0:width] #defino ROI en la zona de las perforacinoes

	#convertir a escala de grices y hacer threshold
	gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
	ROI_height, ROI_width, ROI_channels = ROI.shape

	#imagen vacia en la que voy a poner la ROI y la grafica de intensidad a lo largo del eje x de la ROI
	output_ROI = np.zeros((plot_height+ROI_height,ROI_width,3), np.uint8)
	
	#imagen vacia en la que voy a poner la grafica 
	grafica_ROI = np.zeros((plot_height,ROI_width,3), np.uint8)

	#pego la ROI en la parte superior de la imagen de salida
	output_ROI[0:2*alto_ROI,0:width] = cv2.cvtColor(gray_ROI, cv2.COLOR_GRAY2BGR)
	
	#NOTA: Aparentemente el direccionamiento a los pixels de una imagen es (y,x)

	index = 0
	maximo = 0
	for x in range(0, ROI_width):
		a = 0
		for y in range(0, ROI_height): 
			a = a + gray_ROI[y,x]			#acumular el nivel de intensidad para cada columna
			intensidad[y] = gray_ROI[y,x]	#guardar todos los niveles de intensidad observados en la columna en una lista (para usar la mediana)
		
		#uso la mediana para evitar que los valores extremos afecten el resultado (busco el valor tipico)
		#la mediana resulta aplicable tambien para independizarme del nivel de intensidad que puedan presentar
		#diferentes films y configuraciones de luz
		square_wave[index] = int(np.median(intensidad[0:ROI_height])) 
		
		#registro el valor maximo para normalziar la grafica
		if square_wave[index] > maximo: 
			maximo = square_wave[index]
		index = index + 1

	#Corro de vuelta el mismo loop para graficar
	for i in range(0, ROI_width):
		a = square_wave[i]
		a_norm = a*plot_height/maximo
		cv2.line(grafica_ROI,(i,plot_height),(i, plot_height-a_norm),(0,255,0),2)

	ubicar_bordes_film(img, grafica_ROI)

	#copiar la grafica en la seccion superior de la imagen de salida
	output_ROI[2*alto_ROI:2*alto_ROI+plot_height,0:width] = grafica_ROI

	#dibujar una linea al centro de la ROI y un rectangulo delimitandola
	cv2.line(img,(0,centro_ROI),(width, centro_ROI),(0,255,0),2)
	cv2.rectangle(img,(0,(centro_ROI-alto_ROI)),(width,(centro_ROI+alto_ROI)),(0,0,255),1)
	
	#mostrar las imagenes
	cv2.imshow('image',img)
	cv2.imshow('ROI',output_ROI)
	


def ubicar_bordes_film(img, grafica_ROI):
	borde_superior = 0
	borde_inferior = 0
	#producir imagen binaria para deteccion de los bordes del films
	ret,th1 = cv2.threshold(gray,treshold_borde,255,cv2.THRESH_BINARY_INV)
	
	#busco un salto en las diferencias de intensidad entre pixels consecutivos en la linea central (desde abajo hacia arriba)
	salto = 0; 
	for y in range(d, height):		
		actual = th1[y,int(width/2)].astype(np.int16)
		anterior = th1[y-d,int(width/2)].astype(np.int16)
		derivada = (actual - anterior)/d
		#print "actual = " + str(actual) + " anterior = " + str(anterior) + " derivada = " + str(derivada)
		if derivada > salto:
			borde_superior = y 
			cv2.line(img,(0,borde_superior),(width, borde_superior),(255,0,0),1)
			break

	#busco un salto en las diferencias de intensidad entre pixels consecutivos en la linea central (desde arriba hacia abajo)
	for y in range(height-1, int(height/2), -1):		
		
		actual = th1[height-1,int(width/2)].astype(np.int16)
		anterior = th1[y-1-d,int(width/2)].astype(np.int16)
		derivada = (actual - anterior)/d
		#print "actual = " + str(actual) + " anterior = " + str(anterior) + " derivada = " + str(derivada)
		
		if abs(derivada) > salto:
			borde_inferior = y 
			cv2.line(img,(0,borde_inferior),(width, borde_inferior),(0,255,0),1)
			break

	definir_template(img, borde_superior, borde_inferior)


def definir_template(img, borde_superior, borde_inferior):

	#global px_mm, hole_height, hole_width, hole_position, frame_width, frame_height
	global px_mm, hole_height_px, hole_width_px, hole_position_px, frame_width_px, frame_height_px, inter_hole_px

	height, width, channels = img.shape

	img_template = np.zeros((plot_height,width,3), np.uint8) #imagen vacia para graficar el template

	px_mm = (borde_inferior-borde_superior)/film_height #calculo la relacion mm px a partir del ancho del film detectado

	#calculo el resto de las dimensiones del film a partir de la relacion mm px encontrada antes
	hole_width_px = int(hole_width*px_mm)
	hole_height_px = int(hole_height*px_mm)
	hole_position_px = int(hole_position*px_mm)
	inter_hole_px = int(inter_hole*px_mm)

	frame_width_px = int(frame_width*px_mm)
	frame_height_px = int(frame_height*px_mm)
	
	template = [None for x in range(2*hole_width_px + inter_hole_px)]

	#genero una senal cuadrada con ancho de pulso igual al ancho de agujero 
	for i in range(0, (2*hole_width_px + inter_hole_px)):
		if i < hole_width_px:
			template[i] = 255
		elif (i >= hole_width_px and i <= (hole_width_px + inter_hole_px)):
			template[i] = 0
		elif i > (hole_width_px + inter_hole_px):
			template[i] = 255
		cv2.line(img_template,(i,plot_height),(i, plot_height-template[i]),(0,0,255,200),2)

	correlacion(img, square_wave, template)
	cv2.imshow('template',img_template)



def correlacion(img, square_wave, template):
	correlacion = np.correlate(square_wave,template,'full')
	correlacion_maxima = 0
	for i in range(0, len(correlacion)):
		#print correlacion[i]
		if correlacion[i] > correlacion_maxima:
			correlacion_maxima = correlacion[i]
			frame_x = i

	#print "correlacion maxima " + str(correlacion_maxima)
	#print "posicion " + str(frame_x)

	#cv2.line(img,(frame_x,height),(frame_x, 0),(255,0,255,200),1)

	extraer_frame(img, frame_x, contador_frame)



def extraer_frame(img, frame_x, contador_frame):

	#dibujar linea vertical el el punto de maxima correlacion (bordo derecho del segundo sprocket hole)
	cv2.line(img,(frame_x,height),(frame_x,0),(255,0,0,200),1	)

	#dibujar rectangulo sobre sprocket hole derecho
	cv2.rectangle(img,(frame_x - hole_width_px, hole_position_px),(frame_x,hole_position_px + hole_height_px),(255,0,255),1)

	#dibujar rectangulo sobre sprocket hole izquierdo
	cv2.rectangle(img,(frame_x - 2*hole_width_px - inter_hole_px, hole_position_px),(frame_x - hole_width_px - inter_hole_px,hole_position_px + hole_height_px),(255,0,255),1)

	#calcular las coordenadas del centro del frame a partir de la coordenada x del borde derecho del sprocket hole detectado mediante matching
	#centro_frame_x = int(frame_x) - int(hole_width_px/2) - int(frame_width_px/2)
	centro_frame_x = int(frame_x) - int(hole_width_px) - int(inter_hole_px/2)
	centro_frame_y = borde_superior + int(hole_position_px) + int(hole_height_px) + int(frame_height_px/2)

	#dibujar lineas que se intersecten en el centro del frame detectado
	cv2.line(img,(centro_frame_x,height),(centro_frame_x, 0),(0,255,255),1)
	cv2.line(img,(0,centro_frame_y),(width, centro_frame_y),(0,255,255),1)

	#dibujar rectangulo sobre el frame
	frame_width = inter_hole_px + int(hole_width_px/2)
	cv2.rectangle(img,(centro_frame_x-int(frame_width_px/2),centro_frame_y-int(frame_height_px/2)),(centro_frame_x+int(frame_width_px/2),centro_frame_y+int(frame_height_px/2)),(255,0,255),1)

	frame_detectado = img[centro_frame_y-int(frame_height_px/2):centro_frame_y+int(frame_height_px/2),centro_frame_x-frame_width_px/2:centro_frame_x+int(frame_width_px/2)]
	nombre_archivo_frame = "frame"+str(contador_frame)+".png"
	cv2.imwrite(nombre_archivo_frame,frame_detectado)

cv2.namedWindow('image')
cv2.namedWindow('ROI')
cv2.namedWindow('template')

cv2.createTrackbar('alto_ROI','image',10,255,definir_ROI)
cv2.createTrackbar('centro_ROI','image',40,255,ubicar_centro_ROI)
cv2.createTrackbar('treshold_borde','image',127,255,definir_treshold_borde)
cv2.imshow('image',img)


if(primera_ejecucion == 1):

	while(1):

		if cv2.waitKey(20) & 0xFF == 27:
			break

		alto_ROI = cv2.getTrackbarPos('alto_ROI','image')
		centro_ROI = cv2.getTrackbarPos('centro_ROI','image')
		treshold_borde = cv2.getTrackbarPos('treshold_borde','image')

		f = open('config.obj', 'w') 
		pickle.dump([alto_ROI, centro_ROI, treshold_borde], f)
		f.close()
		print("es primera ejecucion")

	    
	cv2.destroyAllWindows()

else:

	f = open('config.obj', 'r')
	alto_ROI, centro_ROI, treshold_borde = pickle.load(f)
	f.close()
	definir_ROI(alto_ROI)
	print("NO es primera ejecucion")
	cv2.waitKey(0)

	cv2.destroyAllWindows()
		