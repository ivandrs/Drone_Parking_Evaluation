# Drone_Parking_Evaluation
Automated driving - GUI for the visual evaluation of drone data

When developing driver assistance systems and automated driving functions, 
drones can be used to efficiently evaluate vehicle tests. This is done by 
processing image and video recordings of certain test scenarios, for example 
in automated parking attempts. Reference data can be extracted from the image 
and video recordings, which contain the geometric relationships such as distances 
and orientations between the objects involved in the traffic situation - in 
particular the vehicles - and the infrastructure such as lane markings. 
This data helps to evaluate the performance of driver assistance functions.

Json description  
Box = pixels X,Y  de las cuatro esquinas que rodeanlos autos (superior izq, superior  derecha, inferior izq, inferior derecha )
line =pixels X,Y  del inicio y fin de las marcas viales
todas las medidas están en mm
