from ase import build
from ase import io

W = build.make_supercell(build.bulk("W", cubic = True), [[4,0,0],[0,4,0],[0,0,4]])
io.write("W.xyz",W)