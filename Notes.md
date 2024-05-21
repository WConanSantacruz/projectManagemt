#Geometrical speficication
#Given the space based as
P={x,y,z}
And the triangule given by the expresion:
T={P1,P2,P3}
Where P1,P2,P3 are in the real domain, not complex. In the computational Side, we recoment to use the next notation:
Minimal Tol> Can be setted as:

Space for computations
Min=0
Max=0

Regularization of values
Nv=(Point-Middle)/(Max-Min)
Therefore we know that are some adjustamets, like corrections>



The normal, can be computed as:
Middle of triangle
Mid=(P1+P2+P3)/3
Normal computation, will be computing two normal vectors from the middle to the fist two point sets as:
dir1=||(Mid-P1)||
dir2=||(Mid-P2)||
normal=dir1xdir2
That will define the normal vector from the Middle Point.

Where a conectivity is given as the next chainning expresion:
If =Min|P1-P2|=0 the triangules are connected, if not, there are separated. By floating opetration is normal to set a value
as minimun expresed as:
Min|P1-P2|<Tol
Where the points can be adjusted to a middle position, common explained as Merge close vertex, it is common to fix some meshes by the floating point tolerance.

Principal of conectivity:
Also cause each point is shared by an Index, we can define as general expresion T={P1,P2,P3} so therefore a closed mesh, 
can be checked using 3 arguments>
Each Vertex, needs to be at least in 3 triangules.
Each edge, needs to be in two triangles, not more:

Principal of non-invasion:
A general surface can be noted by:
z=Ax+By+C, to check the surface, therefore, it is limited by the P1,P2,P3 surfaces, so, At first, projection principal, 


To check the correct use of the VERTEX, it is common that a vertex is used by a minimun of 3 triangules, that makes a simple way to check if the mesh is closed. 

In a Cube each face is created by two 
To define a fully closed mesh, all the points needs to be conected and also the inverted normals, points sets can not go outside of the mesh:
