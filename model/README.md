# Panel model

`panel.npz` contains a 280-node stiffness matrix, 18-actuator load map, 29-channel observation matrix, coordinates, and 20 gap/gain cases. Each case uses `B = load_map * case_gain`.

DD uses RBF interpolation at all 56 surface targets. MeanRG excludes the 20 fixed boundary nodes; MaxRG uses all 280 nodes. Coordinates and displacements are in mm; forces are in N.
