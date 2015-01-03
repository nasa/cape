#ifndef _PC_TRI_H
#define _PC_TRI_H

PyObject *
pc_WriteTri(PyObject *self, PyObject *args);
char doc_WriteTri[] =
"Write a Cart3D triangulation to :file:`Components.i.tri` file\n"
"\n"
":Call:\n"
"   >>> pc.WriteTri(P, T)\n"
":Inputs:\n"
"   *P*: :class:`numpy.ndarray` (:class:`float`) shape=(N,3)\n"
"       Matrix of nodal coordinates\n"
"   *T*: :class:`numpy.ndarray` (:class:`int`) shape=(M,3)\n"
"       Matrix of of nodal indices for each triangle\n"
":Versions:\n"
"   * 2014-01-02 ``@ddalle``: First version\n";


PyObject *
pc_WriteCompID(PyObject *self, PyObject *args);
char doc_WriteCompID[] =
"Write component ID numbers to :file:`Components.i.tri`\n"
"\n"
":Call:\n"
"   >>> pc.WriteCompID(C)\n"
":Inputs:\n"
"   *C*: :class:`numpy.ndarray` (:class:`int`) shape=(M,)\n"
"       Vector of component IDs\n"
":Versions:\n"
"   * 2014-01-02 ``@ddalle``: First version\n";

#endif
