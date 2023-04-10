# Proyecto de Optimización - Rodrigo Castillo Camargo


# Enunciado:
De acuerdo con la guía de la asignatura, consulte en el texto “M. Bazaraa, H. Sherali, C. Shetty. Nonlinear
Programming: Theory and Algorithms. Wiley-Interscience, 3rd ed. 2006”, los numerales 8.5, 8.6, 8.7 y 8.8.
Revise especialmente los siguientes métodos

1. Método de descenso escalonado
2. Método de Fletcher y Reeves
3. Método de Davidon-Fletcher-Powell. 
4. Método de Broyden, Fletcher, Coldfarb-Shanno.

Posteriormente, plantee dichos algoritmos para la solución del siguiente problema de optimización no
restringido:

<!--Enunciado del problema de optimización lineal: --> 
$$
min (x_1 - x_2^3) + 3(x_1 - x_2)^4
$$

## Trabajo a Realizar
1. Programar dichos métodos en cualquier lenguaje de programación
2. Resuelva y responda: ¿Convergen en el mismo punto todos los algoritmos? Si no es así, analice del porqué de la situación presentada.
3. Si se restringe el problema de la siguiente manera:
$$
min (x_1 - x_2^3) + 3(x_1 -x_2) ^4
$$
sujeto a:
$$
-x_1 + x_2  =1 
$$
$$
x_1 + x_2 \leq 2
$$
$$
x_1,x_2 \geq 0
$$

- Resuélvalo por cualquier método, programando un algoritmo en cualquier lenguaje para su
solución. ¿Tiene el PNL restringido solución óptima? analice y explique sus respuestas
- Realice una presentación en la que se explique claramente la solución presentada a cada punto. Anexe los archivos y/o soportes necesarios que justifiquen la solución dada

--- 

# Solución

## Observaciones generales
La función generada por la función $(x_1 - x_2^3) + 3(x_1 - x_2)^4$ acotada por las restricciones

1. $-x_1 + x_2  =1$
2. $x_1 + x_2 \leq 2$
3. $x_1 \geq 0$
4. $x_2 \geq 0$

luce así

![graphic2](img/graphic2.png)

## Programación de los métodos

### 1. Método de descenso escalonado
#### Introducción: 
El método del descenso escalonado, también conocido como método simplex, es un algoritmo utilizado en la optimización lineal para encontrar la solución óptima de un problema de programación lineal. 

El método del descenso escalonado comienza con una solución inicial y luego realiza iteraciones para mejorarla. En cada iteración, se elige una variable no básica (es decir, una variable que no es igual a cero en la solución actual) y se intenta aumentarla o disminuirla para mejorar la función objetivo)

El algoritmo funciona moviendo la solución actual de esquina a esquina del poliedro de soluciones factibles, siempre mejorando la función objetivo en cada paso, hasta que se alcanza una solución óptima.

El nombre "descenso escalonado" se refiere a cómo el algoritmo mueve de una esquina a otra del poliedro de soluciones factibles. En cada iteración, se mueve a una esquina adyacente al "escalón" más cercano, lo que lleva a una reducción en el valor de la función objetivo



### 2. Método de Fletcher y Reeves

### 3. Método de Davidon-Fletcher-Powell. 

### 4. Método de Broyden, Fletcher, Coldfarb-Shanno.

## Evaluaciones de los métodos.
