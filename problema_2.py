import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Leo imagen --------------------------------------------------
img = cv2.imread('TP 1/formulario_vacio.png',cv2.IMREAD_GRAYSCALE)

plt.figure()                  
plt.imshow(img, cmap='gray')  
plt.show()    

th_value = 200
img_th = (img < th_value).astype(np.uint8) * 255  #Luego borrar este comentario: si encuentra algo gris/negro (<200) lo multiplica * 255, al pasarse de 255, el astype int 8 le hace de tope

plt.figure()                  
plt.imshow(img_th, cmap='gray')  
plt.show()    

img_cols = np.sum(img_th, axis=0) / 255
img_rows = np.sum(img_th, axis=1) / 255

th_col = img_cols.max()*0.95
th_row = 230

rows_with_lines = img_rows > th_row
cols_with_lines = img_cols > th_col


def get_line_positions(line_detection_array):
    positions = []
    is_line = False
    start = 0
    for i, val in enumerate(line_detection_array):
        if val and not is_line:
            is_line = True
            start = i
        elif not val and is_line:
            is_line = False
            # Se toma el punto medio de la línea detectada
            positions.append(start + (i - 1 - start) // 2)
    return positions

horizontal_lines = get_line_positions(rows_with_lines)
vertical_lines = get_line_positions(cols_with_lines)

plt.figure(figsize=(15, 5))


"""
SOLUCIÓN ANTERIOR, DIBUJABA LAS LINEAS DE PUNTA A PUNTA --  VER DE MEJORARLA

img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for y in horizontal_lines:
    cv2.line(img_with_lines, (0, y), (img.shape[1], y), (0, 255, 0), 2)

for x in vertical_lines:
    cv2.line(img_with_lines, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)

cv2.imshow('Líneas Detectadas', img_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for y in horizontal_lines:
    for i in range(len(vertical_lines) -1):
        x1 = vertical_lines[i]
        x2 = vertical_lines[i+1]
        cv2.line(img_with_lines, (x1, y), (x2, y), (0, 255, 0), 2)
        
for x in vertical_lines:
    # Para cada línea vertical, dibujamos un segmento por cada "fila"
    # Iteramos hasta la penúltima línea horizontal para tener siempre un par (inicio, fin)
    for i in range(len(horizontal_lines) - 1):
        y1 = horizontal_lines[i]
        y2 = horizontal_lines[i+1]
        
        # Dibujamos el segmento de línea vertical desde y1 hasta y2 en la posición x
        cv2.line(img_with_lines, (x, y1), (x, y2), (0, 0, 255), 2)

cv2.imshow('Celdas Delimitadas', img_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()