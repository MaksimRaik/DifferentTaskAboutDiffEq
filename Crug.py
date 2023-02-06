import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numba import jit

h = 0.2
tau = 0.2
N = 100

phi = 2. * np.pi / (N - 1)

@jit
def StartValue( h, phi ):

    global N

    PHI = np.arange( 0, 2. * np.pi + phi, phi )

    R = np.arange( 0, N * h, h )

    u = np.zeros( ( N, N ) )

    return u, R, PHI

@jit
def Rplus( r, m ):

    global phi

    return 2.0 * ( r[ m + 1 ] * r[ m ] ) / ( r[ m + 1 ] + r[ m ] )

@jit
def Rminus( r, m ):

    return 2.0 * ( r[ m ] * r[ m - 1 ] ) / ( r[ m ] + r[ m - 1 ] )

@jit
def funG( t ):

    return 1. #3. + 7. * t

@jit
def ABCD( r ):
    global N, Rplus, Rminus
    global h, phi

    A = np.zeros( ( N, N ) )
    B = np.zeros( ( N, N ) )
    C = np.zeros( ( N, N ) )
    D = np.zeros( ( N, N ) )
    F = np.zeros( ( N, N ) )

    for i in np.arange(1, N - 1, 1):

        for j in np.arange(1, N, 1):

            A[ i ][ j ] = Rplus( r, i ) * r[ i ] / h ** 2 / ( Rplus( r, i ) * r[ i ] / h ** 2 + Rminus( r, i ) * r[ i ] / h ** 2 + 2. / phi ** 2 )

            B[ i ][ j ] = Rminus( r, i ) * r[ i ] / h ** 2 / ( Rplus( r, i ) * r[ i ] / h ** 2 + Rminus( r, i ) * r[ i ] / h ** 2 + 2. / phi ** 2 )

            C[ i ][ j ] = 1. / phi ** 2 / ( Rplus( r, i ) * r[ i ] / h ** 2 + Rminus( r, i ) * r[ i ] / h ** 2 + 2. / phi ** 2 )

            D[ i ][ j ] = 1. / phi ** 2 / ( Rplus( r, i ) * r[ i ] / h ** 2 + Rminus( r, i ) * r[ i ] / h ** 2 + 2. / phi ** 2 )

            F[ i ][ j ] = r[ i ] ** 2 / ( Rplus( r, i ) * r[ i ] / h ** 2 + Rminus( r, i ) * r[ i ] / h ** 2 + 2. / phi ** 2 )

    return A, B, C, D, F

@jit
def Iter( u, R, T ):

    global ABCD, funG
    global N

    a, b, c, d, f = ABCD( R )

    uNew = np.zeros( ( N, N ) )

    for i in np.arange( 1, N - 1, 1 ):

        u[i][0] = a[i][0] * u[i + 1][0] + b[i][0] * u[i - 1][0] + c[i][0] * u[i][1] + \
                     d[i][0] * u[i][N - 1] + funG(T) * f[i][0]

        for j in np.arange( 1, N - 1, 1 ):

            u[ i ][ j ] = a[ i ][ j ] * u[ i + 1 ][ j ] + b[ i ][ j ] * u[ i - 1 ][ j ] + c[ i ][ j ] * u[ i ][ j + 1 ] +\
                          d[ i ][ j ] * u[ i ][ j - 1 ] + funG( T ) * f[ i ][ j ]

        u[i][N-1] = a[i][N-1] * u[i + 1][N-1] + b[i][N-1] * u[i - 1][N-1] + c[i][N-1] * u[i][0] + \
                     d[i][N-1] * u[i][N - 2] + funG(T) * f[i][N-1]

    return u

@jit
def IterCalc( u, R ):

    global N, tau
    global Iter

    uOld = np.ones( ( N, N ) ) * 100.

    for tt in np.arange( 0, 10000000, 1 ):

        if np.linalg.norm( uOld - u ) < 1.0e-3 or tt > 200000:

            break

        uOld = np.copy( u )

        u = Iter( u, R, tt * tau )

    return u

u, R, PHI = StartValue( h, phi )

uNew = IterCalc( u, R )

a = R[ -1 ]

R, PHI = np.meshgrid( R, PHI )

y = R * np.sin( PHI )

z = R * np.cos( PHI )

plt.figure( figsize=( 20, 10 ) )
plt.subplot( 211 )
plt.grid()
plt.xlabel( 'Z' )
plt.ylabel( 'Y' )
plt.contourf( z, y, uNew, cmap = 'hot' )
plt.colorbar()
plt.subplot( 212 )
plt.grid()
plt.xlabel( 'Z' )
plt.ylabel( 'Y' )
plt.contourf( z, y, 1./4. * ( a ** 2 - z ** 2 - y ** 2 ), cmap = 'hot' )
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d' )
ax.plot_surface( z, y, uNew, cmap = 'hot' )
plt.show()








