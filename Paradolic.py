import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import animation

#Constant

L = 2.
h = 0.1
tau = 0.0002
lam0 = 0.5
sigma = 2.
c = 1.0
u0 = ( sigma * c ** 2 / lam0 ) ** ( 1. / sigma )
gamma = 0.5 + h ** 2 / ( 12. * lam0 * tau )

x = np.arange( 0, L, h )
U0 = np.zeros( x.size ) + 0.0001


@jit
def Lam( u ): # lambda( u ) = lambda_0 * u^k

    global sigma, lam0

    return lam0 * u ** sigma

@jit
def LamFun( uPlus, uMinus ): # lambda_+/- = ( lambda_m + lambda_{m+1/m-1} ) / 2.

    global Lam

    LamPlus = Lam( uPlus )

    LamMinus = Lam( uMinus )

    return ( LamPlus + LamMinus ) / 2.

@jit
def Approx( u, m ):

    global LamFun
    global h

    LamPlus = LamFun( u[ m + 1 ], u[ m ] )

    LamMinus = LamFun( u[ m ], u[ m - 1 ] )

    return ( LamPlus * ( u[ m + 1 ] - u[ m ] ) - LamMinus * ( u[ m ] - u[ m - 1 ] ) ) / h ** 2

@jit
def A( u, m ):

    global h, gamma, tau
    global LamFun

    return gamma * tau * LamFun( u[ m ], u[ m - 1 ] ) / h ** 2

@jit
def B( u, m ):

    global h, gamma, tau
    global LamFun

    LamPlus = LamFun( u[ m ], u[ m + 1 ] )

    LamMinus = LamFun( u[ m ], u[ m - 1 ] )

    return gamma * tau * ( LamPlus + LamMinus ) / h ** 2 + 1.

@jit
def C( u, m ):

    global h, gamma, tau
    global LamFun

    return gamma * tau * LamFun( u[ m ], u[ m + 1 ] ) / h ** 2

@jit
def D( u, m ):

    global h, gamma, tau
    global LamFun, Approx

    return u[ m ] + tau * ( 1. - gamma ) * Approx( u, m )

@jit
def AlphaBetta( u, m, alpha_old, betta_old ):

    global A, B, C, D

    alphaNext = C( u, m ) / ( B( u, m ) - A( u, m ) * alpha_old )

    bettaNext = ( A( u, m ) * betta_old + D( u, m ) ) / ( B( u, m ) - A( u, m ) * alpha_old )

    return alphaNext, bettaNext

@jit
def AlBt( u, t ):

    global AlphaBetta, sigma, u0

    alfa = np.zeros( u.size )
    beta = np.zeros( u.size )

    alfa[1] = 0.0

    beta[1] = u0 * t ** (1. / sigma)

    for i in np.arange( 2, u.size, 1 ):

        alfa[ i ], beta[ i ] = AlphaBetta( u, i - 1, alfa[ i - 1 ], beta[ i - 1 ] )

    return alfa, beta

@jit
def SixPointsMth( u, T ):

    global AlBt
    global h, tau, gamma, u0

    uNew = np.copy(u)

    M = int( T / tau )

    for i in np.arange( 0, M, 1 ):

        t = i * tau

        alpha, betta = AlBt( uNew, t )

        uNew[ -1 ] = 0.0

        for m in np.arange( u.size - 2, 0, -1 ):

            uNew[ m ] = alpha[ m + 1 ] * uNew[ m + 1 ] + betta[ m + 1 ]

        uNew[0] = u0 * t ** (1. / sigma)

    return uNew

@jit
def Evid( u, T ):

    global h, tau, gamma, u0
    global Approx

    uNew = np.copy( u )

    M = int( T / tau )

    for i in np.arange( 0, M, 1 ):

        uNew[ 0 ] = u0 * ( i * tau ) ** (1. / sigma)

        uNew[ -1 ] = 0.0

        for j in np.arange( 1, u.size - 1, 1 ):

            uNew[ j ] = uNew[ j ] + tau * Approx( uNew, j )

    return uNew


@jit
def SurfU0( u, T ):

    global SixPointsMth, Evid
    global tau

    N = x.size

    M = int( T / tau )

    UU = np.zeros( ( M, N ) )

    U = np.copy( u )

    for n in np.arange( 1, M, 1 ):

            UU[ n ] = Evid( U, n *tau )

    return UU

@jit
def Toch( x, T ):

    global c, sigma, h, lam0

    u = np.zeros( x.size )

    for m in np.arange( 0, u.size, 1 ):

        if m * h > c * T:

            u[ m ] = 0.0

        else:

            u[ m ] = ( sigma * c * ( c * T - m * h ) / lam0 ) ** ( 1. / sigma )

    return u

u11 = Toch( U0, 1. )
u12 = Evid( U0, 1. )
u13 = SixPointsMth( U0, 1 )

u21 = Toch( U0, .5 )
u22 = Evid( U0, .5 )
u23 = SixPointsMth( U0, 0.5 )

u31 = Toch( U0, 1.5 )
u32 = Evid( U0, 1.5 )
u33 = SixPointsMth( U0, 1.5 )

plt.figure( figsize = ( 15, 10 ) )
plt.rc('font', **{'size' : 20})
plt.grid()
plt.xlabel( 'X' )
plt.ylabel( 'U' )
plt.plot( x, u11,
          x, u12,
          x, u13,)

plt.plot( x, u21,
          x, u22,
          x, u23,)

plt.plot( x, u31,
          x, u32,
          x, u33,)

plt.legend( [ 'tochnoe t = 1.' , 'evid t = 1.', 'sixmt t = 1.',
              'tochnoe t = 0.5', 'evid t = 0.5', 'sixmt t = 0.5',
              'tochnoe t = 1.5', 'evid t = 1.5', 'sixmt t = 1.5' ] )
plt.show()














