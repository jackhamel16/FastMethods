import numpy as np
import numpy.linalg as lg
import tree
import interaction
import utilities as utils




test = interactions.uv_list
Gt = np.dot(test[16][31][0], test[16][31][1])
test2 = my_tree.tree
G = interactions.build_G(test2[16],test2[31])
src_vect = interactions.src_vecs[16][31]
src_vec = np.array([src_list[i].weight for i in my_tree.tree[31]])

B = np.dot(G, src_vec)
Bt = np.dot(Gt, src_vect)












#eps = 1e-1
#
#G1, G2, G3, G4 = G[0:4,0:4], G[4:,0:4], G[0:4,4:], G[4:,4:]
#U1, V1 = utils.uv_decompose(G1, eps)
#U2, V2 = utils.uv_decompose(G2, eps)
#U3, V3 = utils.uv_decompose(G3, eps)
#U4, V4 = utils.uv_decompose(G4, eps)
#
#U12,V12 = utils.merge(U1,V1,U2,V2,eps)
#U34,V34 = utils.merge(U3,V3,U4,V4,eps)
#
#U, V = utils.merge(U12,V12,U34,V34,eps,1)
#Bt = np.dot(U, V)
#
#error = (lg.norm(Bt) - lg.norm(G)) / lg.norm(G)
#print(error)














