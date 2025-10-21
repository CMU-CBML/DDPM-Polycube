#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

d = np.array(np.loadtxt("../context.txt","int"))


v_co_torus =  np.loadtxt(r'../v_cos/512_torus_h.txt')
v_co_cube =  np.loadtxt(r'../v_cos/512_cube.txt')
v_co_torus_x_h =  np.loadtxt(r'../v_cos/512_torus_h_1x_oriented_hole.txt')
v_co_torus_y_h =  np.loadtxt(r'../v_cos/512_torus_h_1y_oriented_hole.txt')

v_cos = [v_co_cube,v_co_torus,v_co_torus_x_h,v_co_torus_y_h]


def normalize2(x):
    _max = np.max(x)
    _min = np.min(x)
    return ((x - (_max + _min) / 2) / (_max - _min) * 2)

def standardize2(x):
    x = x.reshape(-1,3)
    return ((x - np.mean(x,axis = 0)) / np.std(x,axis = 0))

def sortby_x(arr):
    seq = np.argsort(arr[:,0])
    #print(seq[:5])
    return seq,arr[seq]

#print(sortby_x(normalize2(standardize2(v_co_torus_y_h))))

seq_cube,v_co_cube = sortby_x(v_co_cube)
seq_torus,v_co_torus = sortby_x(v_co_torus)
seq_torus_x_h,v_co_torus_x_h = sortby_x(v_co_torus_x_h)
seq_torus_y_h,v_co_torus_y_h = sortby_x(v_co_torus_y_h)


reverse_seq_cube = np.argsort(seq_cube)
reverse_seq_torus = np.argsort(seq_torus)
reverse_seq_torus_x_h = np.argsort(seq_torus_x_h)
reverse_seq_torus_y_h = np.argsort(seq_torus_y_h)

r_seqs = [reverse_seq_cube,reverse_seq_torus,reverse_seq_torus_x_h,reverse_seq_torus_y_h]


v_id_cube =  np.loadtxt(r'../v_cos/v_id_per_face_polygon_cube.txt').astype('int')
v_id_torus = np.loadtxt(r'../v_cos/v_id_per_face_polygon_torus.txt').astype('int')
v_id_torus_x_h = np.loadtxt(r'../v_cos/v_id_per_face_polygon_torus_x_hole.txt').astype('int')
v_id_torus_y_h = np.loadtxt(r'../v_cos/v_id_per_face_polygon_torus_y_hole.txt').astype('int')

v_ids = [v_id_cube,v_id_torus,v_id_torus_x_h,v_id_torus_y_h]


v_id = 0

if d[0] != -1:
    #v_id_1 = v_id_cube + 1
    #v_id_2 = v_id_torus + 513
    v_id_1 = v_ids[d[0] // 2 ] + 1
    print(v_id_1.shape)
    v_id_2 = v_ids[d[1] // 2 ] + 513
    print(v_id_2.shape)
    v_id = np.vstack([v_id_1,v_id_2])

else:
    v_id = v_ids[d[0] // 2]
    
cat = np.ones(len(v_id))


# In[2]:


import numpy as np
import os



pattern1 = ['$# LS-DYNA Keyword file created by LS-PrePost(R) V4.5.3 - 28Oct2017',
'$# Created on Mar-15-2018 (10:40:36)',
'*KEYWORD',
'*ELEMENT_SHELL']
pattern2 = ['*NODE']
pattern3 = ['*END']



faces_info = np.zeros((len(cat),10)).astype('int')
faces_info[:,0] = np.arange(1,len(cat) + 1)

faces_info[:,1] = cat
faces_info[:,2:5] = v_id
faces_info[:,5] = v_id[:,2]

faces_info = faces_info.astype('str')

for i in range(faces_info.shape[0]):
    for j in range(faces_info.shape[1]):
        faces_info[i][j] = faces_info[i][j].rjust(8)
        
np.savetxt('../tmp/faces_info.txt',faces_info,'%s',delimiter='')  

#import the result of prediction
v_cos = np.loadtxt(r"../tmp/case0.txt").reshape(33,3,32,32)

for t in range(len(v_cos)):
    
    v_co = v_cos[t].transpose([1,2,0]).reshape(-1,3)

    if d[0] == -1:
        v_co = v_co[512:][r_seqs[d[1] // 2]]
    else:
        v_co[:512] = v_co[:512][r_seqs[d[0] // 2]]
        v_co[512:] = v_co[512:][r_seqs[d[1] // 2]]
    v_co = np.around(v_co, decimals=5)  


    p1 = (np.arange(1,len(v_co) + 1)).astype('int').reshape(-1,1).astype('str')
    p2 = v_co.astype('str')
    p3 = np.zeros((len(v_co),2)).astype('int').astype('str')

    vertex_info = np.hstack((np.hstack((p1,p2)),p3))

    for i in range(vertex_info.shape[0]):
        for j in range(vertex_info.shape[1]):
            vertex_info[i][j] = vertex_info[i][j].rjust(8) if (j == 0 or j == 4 or j == 5) else vertex_info[i][j].rjust(16)

    np.savetxt('../tmp/vertices_info.txt',vertex_info,fmt = '%s',delimiter = '')


    with open('../tmp/vertices_info.txt','r') as v_info:
        v_i = v_info.readlines()
    with open('../tmp/faces_info.txt','r') as f_info:
        f_i = f_info.readlines()


    #get k files
    
    output_file_name = f"../diff_32_steps_stl/step{t}.k"
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    if not os.path.exists(output_file_name):
        file = open(output_file_name, "w")
        file.close()

    with open(output_file_name,'a+') as f:
        for i in range(len(pattern1)):
            f.write(pattern1[i])
            f.write('\n')
        for i in range(len(f_i)):
            f.write(f_i[i])
        for i in range(len(pattern2)):
            f.write(pattern2[i])
            f.write('\n')
        for i in range(len(v_i)):
            f.write(v_i[i])        
        for i in range(len(pattern3)):
            f.write(pattern3[i])
            f.write('\n')
    
    
    #get stl files based on k files
        
    with open(output_file_name) as f:
        p = f.readlines()
    output_file_name = f"../diff_32_steps_stl/step{t}.stl"
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    if not os.path.exists(output_file_name):
        file = open(output_file_name, "w")
        file.close()


    p = np.array(p)
    indices = np.arange(len(p))

    flag1 = indices[p == '*ELEMENT_SHELL\n',][0]
    flag2 = indices[p == '*NODE\n',][0]
    flag3 = indices[p == '*END\n',][0]

    faces = p[flag1+1:flag2]
    face_ver = []
    for i in range(len(faces)):
        q = np.array(faces[i].split(' '))
        face_ver.append(q[q != ''][2:5].astype('int'))
    face_ver = np.array(face_ver) - 1


    vertices = p[flag2 + 1:flag3]
    vert_co = []
    for i in range(len(vertices)):
        q = np.array(vertices[i].split(' '))
        vert_co.append(q[q != ''][1:4].astype('float'))

    vert_co = np.array(vert_co)    

    vert_id = []    
    for i in range(len(vertices)):
        q = np.array(vertices[i].split(' '))
        vert_id.append(q[q != ''][0].astype('int'))
    vert_id = np.array(vert_id)
    vert_id = vert_id - 1

    
    new_id = np.arange(len(vert_id))
    d = dict(zip(vert_id,new_id))

    face_ver =  np.vectorize(d.get)(face_ver)

    def get_normal(v):
        p1 = v[0] - v[1]
        p2 = v[0] - v[2]
        p3 = np.cross(p1,p2)
        p3 = p3 / (np.sum(p3 ** 2)) ** 0.5
        return p3

    normals = []
    for i in range(len(face_ver)):
        normals.append(get_normal(vert_co[face_ver[i]]))
    normals = np.array(normals)
    normals = np.around(normals,6)

    with open(output_file_name,"a+") as f:
        f.write("solid Exported from Blender-3.0.0\n")
        for i in range(len(face_ver)):
            f.write("facet normal")
            for j in range(3):
                f.write(' ')
                f.write(str(normals[i][j]))
            f.write('\n')
            f.write('outer loop\n')
            for j in range(3):
                f.write("vertex")
                for k in range(3):
                    f.write(' ')
                    f.write(str(vert_co[face_ver[i]][j][k]))
                f.write('\n')
            f.write('endloop\n')
            f.write('endfacet\n')


# In[ ]:




