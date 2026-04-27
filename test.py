import scipy.io as sio
import scipy.sparse as ssp
import ssgetpy
import os

def download_matrix(destpath='data/', format='MAT'):
    A = ssgetpy.search(limit=1)[0]
    print(A)
    A.download(format=format, destpath=destpath)
    return os.path.join(destpath, A.name + ('.mat' if format == 'MAT' else '.tar.gz'))

def load_matrix(path):
    mat = sio.loadmat(path)
    A = mat['Problem']['A'][0, 0]
    return ssp.csc_matrix(A)

if __name__ == "__main__":
    path = download_matrix()
    print(path)
    A = load_matrix(path)
    print(A)
