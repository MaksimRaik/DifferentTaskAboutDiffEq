import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from timeit import*

### Constant

L = 100.
T = 0.2
gamma = 5. / 3.

sigma = 0.01
h = 0.2
tau = sigma * h

vr, vl = 0.0, 0.0
rhor, rhol = 1.3, 13.
pr, pl = 1.e5, 10.e5

start = default_timer()

@jit
def StartValue( L, h ):

    global vr, vl
    global rhor, rhol
    global pr, pl
    global gamma

    x = np.arange(-L, L, h)

    rho = np.zeros( x.size )
    u = np.zeros( x.size )
    e = np.zeros( x.size )
    p = np.zeros( x.size )

    for i in np.arange( 0, x.size, 1 ):

        if x[ i ] <= 0.0:

            rho[ i ] = rhol

            p[ i ] = pl

            e[ i ] = pl / (gamma - 1.) / rhol

        else:

            rho[ i ] = rhor

            p[ i ] = pr

            e[ i ] = pr / (gamma - 1.) / rhor

    return x, rho, p, u, e


@jit
def matrixA( u, e ):

    global gamma

    A = np.asarray( [ [ 0., 1., 0. ],
                      [ - u ** 2, 2. * u, gamma - 1. ],
                      [ - u * e * gamma, gamma * e, u ] ] )

    return A

@jit
def matrixOmegaT( u, c ):

    global gamma

    OmegaT = np.asarray( [ [ -u * c, c, gamma - 1. ],
                      [ - c ** 2, 0., gamma - 1. ],
                      [ u * c, -c, gamma - 1. ] ] )

    return OmegaT

@jit
def matrixLam( u, c ):

    Lam = np.zeros( ( 3, 3 ) )

    Lam[ 0 ][ 0 ] = u + c
    Lam[ 1 ][ 1 ] = u
    Lam[ 2 ][ 2 ] = u - c

    return Lam

@jit
def KIR( wPlus, w, wMinus, OmegaT, A, Lam, sigma ):

    global gamma

    OmOm1 = np.dot( np.linalg.inv( OmegaT ), np.fabs( Lam ) )

    OmOm = np.dot( OmOm1, OmegaT )

    return w - np.dot( A, ( wPlus - wMinus ) ) * sigma / 2. + np.dot( OmOm, (wPlus - 2. * w + wMinus) ) * sigma / 2.

@jit
def MatrixCalc( u, e ):

    global  gamma
    global matrixA, matrixLam
    global matrixOmegaT

    N = u.size

    A = np.zeros( ( N, 3, 3 ) )
    OmegaT = np.zeros( ( N, 3, 3 ) )
    Lamda = np.zeros( ( N, 3, 3 ) )
    c = np.zeros( N )

    for i in np.arange( 0, N ,1 ):

        A[ i ] = matrixA( u[ i ], e[ i ] )

        c[ i ] = np.sqrt( gamma * ( gamma - 1. ) * e[ i ] )

        OmegaT[ i ] = matrixOmegaT( u[ i ], c[ i ] )

        Lamda[ i ] = matrixLam( u[ i ], c[ i ] )

    return A, OmegaT, Lamda

@jit
def Iter( rho, u, p, e, h, sigma, T ):

    global KIR, gamma
    global MatrixCalc

    tau = sigma * h

    j = 1

    while tau * j < T:

        j = j + 1

        w = np.zeros( ( u.size, 3 ) )

        w[ :, 0 ] = rho

        w[ :, 1 ] = rho * u

        w[ :, 2 ] = rho * e

        w_new = np.copy( w )

        mxA, mxOmT, mxLam = MatrixCalc( u, e )

        if tau * np.max(mxLam) / h > 1.:

            while tau * np.max(mxLam) / h >= 1.:

                tau = tau / 2.

            sigma = tau / h

        for i in np.arange( 1, u.size - 1, 1 ):

            w_new[ i ] = KIR( w[ i + 1 ], w[ i ], w[ i - 1 ], mxOmT[ i ], mxA[ i ], mxLam[ i ], sigma )

        rho = w_new[ :, 0 ]
        u = w_new[ :, 1 ] / rho
        e = w_new[ :, 2 ] / rho

        rho[ 0 ] = rho[ 1 ]
        u[ 0 ] = u[ 1 ]
        e[ 0 ] = e[ 1 ]

        rho[ -1 ] = rho[ -2 ]
        u[ -1 ] = u[ -2 ]
        e[ -1 ] = e[ -2 ]

        p = (gamma - 1) * rho * e

    return rho, u, p, e

x, rho, p, u, e = StartValue( L, h )

rho, u, p, e = Iter( rho, u, p, e, h, sigma, T )

T1 = 0.005

x1, rho1, p1, u1, e1 = StartValue( L, h )

rho1, u1, p1, e1 = Iter( rho1, u1, p1, e1, h, sigma, T1 )

U = np.asarray( [ [ rho, u, p, e ], [ 'Density for T = '+ str( T ), 'Speed for T = '+ str( T ), 'Pressure for T = '+ str( T ), 'Energy for T = '+ str( T ) ] ] )

U1 = np.asarray( [ [ rho1, u1, p1, e1 ], [ 'Density for T = '+ str( T1 ), 'Speed for T = '+ str( T1 ), 'Pressure for T = '+ str( T1 ), 'Energy for T = '+ str( T1 ) ] ] )

stop = default_timer()
total_time = stop - start

mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print('time work=' + str(hours) + 'h ' + str(mins) + 'min ' + str(round(secs, 1)) + 'sec')

plt.figure( figsize = ( 16, 20 ) )
for j in np.arange( 0, 4, 1 ):

    plt.subplot( 4, 1, j + 1 )
    plt.grid()
    plt.plot( x, U[ 0 ][ j ],  label = U[ 1 ][ j ] )
    plt.plot( x1, U1[ 0 ][ j ], label = U1[ 1 ][ j ] )
    plt.legend()

plt.show()





















