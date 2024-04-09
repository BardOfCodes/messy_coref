# CSG 3D Language

### PreFix Notation

1) Transforms:
   1) Translate (-1, 1)
   2) Rotate (-180, 180) -> (-1, 1)
   3) Scale (0.25, 2.25) -> ()
2) Primitives:
   1) Sphere
   2) Cuboid
   3) Ellipsoid
   4) Cylinder
3) Booleans:
   1) Union
   2) Intersection
   3) Difference


## Program Size: 

Lets Assume 4 tokens for each primitive definition - (type, x, y, z).
Similarly for transform - (type, x, y, z)
boolean  just 1 - (type)

The minimum length of program 