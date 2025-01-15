import numpy as np

# pos_init = np.array([1, 1, 1])

def lorenzAttractor(x, y, z, beta = 2.67, rho = 28, sigma = 10):
    dx = sigma*(y-x)
    dy = (x*(rho-z)-y)
    dz = x*y - beta*z
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.1, 1, 0])

def langfordAttractor(x, y, z, a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1):
    dx = (z-b)*x - d*y
    dy = d*x + (z-b)*y
    dz = c + a*z - (z**3)/3 - (x**2+y**2)*(1 + e*z) + (f*z*x**3)
    return(np.array([dx, dy, dz]))

# pos_init = np.array([1, 1, 1])

def rosslerAttractor(x, y, z, a = 0.2, b = 0.2, c = 5.7):
    dx = -y-z
    dy = x + a*y 
    dz = b+z*(x-c)
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.1, 0.1, 0.1])

def thomasAttractor(x, y, z, a = 0.208186):
    dx = -a*x + np.sin(y)
    dy = -a*y + np.sin(z)
    dz = -a*z + np.sin(x)
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.1, 0.1, 0.1])

def chuaAttractor(x, y, z, alpha = 9, beta = 14.286, m0 = -1.143, m1 = -0.714):
    h = m1*x + 0.5*(m0-m1)*(np.abs(x+1)-np.abs(x-1))
    
    dx = alpha*(y-x-h)
    dy = x-y+z
    dz = -beta*y
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.1, 0.1, 0.1])

def henonHeilesAttractor(x, y, z):
    dx = y
    dy = -x-2*x*z
    dz = -z-x**2+z**2
    return(np.array([dx, dy, dz]))

def halvorsenAttractor(x, y, z, alpha = 1.4):
    dx = -alpha*x - 4*y - 4*z - y**2
    dy = -alpha*y - 4*z - 4*x - z**2
    dz = -alpha*z - 4*x - 4*y - x**2
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.1, 1, 0])

def dadrasAttractor(x, y, z, a = 3, b = 2.7, c = 1.7, d = 2, e = 9):
    dx = y - a*x + b*y*z
    dy = c*y - x*z + z
    dz = d*x*y - e*z
    return(np.array([dx, dy, dz]))

# dt = 0.00001
# pos_init = np.array([5, 10, 10])

def chenLeeAttractor(x, y, z, alpha = 5, beta = -10, delta = -0.38):
    dx = alpha*x - y*z
    dy = beta*y + x*z
    dz = delta*z + (x*y)/3
    return(np.array([dx, dy, dz]))

# pos_init = np.array([1, 0, 4.5])

def rucklidgeAttractor(x, y, z, k = -2, l = -6.7):
    dx = k*x - l*y - y*z
    dy = x
    dz = -z + y**2
    return(np.array([dx, dy, dz]))

# pos_init = np.array([1, 1, 1])

def lorenz83Attractor(x, y, z, a = 0.95, b = 7.91, f = 4.83, g = 4.66):
    dx = -a*x - y**2 - z**2 + a*f
    dy = -y + x*y - b*x*z + g
    dz = -z + b*x*y + x*z
    return(np.array([dx, dy, dz]))

# pos_init = np.array([1, 1, 1])

def rabinovichFabrikantAttractor(x, y, z, alpha = 0.14, gamma = 0.1):
    dx = y*(z-1+x**2) + gamma*x
    dy = x*(3*z + 1 -x**2) + gamma*y
    dz = -2*z*(alpha + x*y)
    return(np.array([dx, dy, dz]))

# dt = 0.0001
# pos_init = np.array([-0.29, -0.25, -0.59])

def threeScrollAttractor(x, y, z, a = 32.48, b = 45.84, c = 1.18, d = 0.13, e = 0.57, f = 14.7):
    dx = a*(y-x) + d*x*z
    dy = b*x - x*z + f*y
    dz = c*z + x*y - e*x**2
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.63, 0.47, -0.54])

def sprottAttractor(x, y, z, a = 2, b = 2):
    dx = y + (a*x*y) + (x*z)
    dy = 1 - (b*np.power(x,2)) + (y*z)
    dz = x - (np.power(x,2)) - (np.power(y,2))
    return(np.array([dx, dy, dz]))

# pos_init = np.array([1, -1, 1])

def fourWingAttractor(x, y, z, a = 1, b = 1, m = 1):
    dx = a*x + y + y*z
    dy = -x*z + y*z
    dz = -z - m*x*y + b
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0.0138, 0, -0.0138])

def sprottCaseAAttractor(x, y, z):
    dx = y
    dy = -x + y*z
    dz = 1 - np.power(y,2)
    return(np.array([dx, dy, dz]))

# pos_init = np.array([0, 0, 0])

def wangChenAttractor(x, y, z):
    dx = y*z + 0.006
    dy = np.power(x,2) - y
    dz = 1 - 4*x
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def boualiAttractor(x, y, z, alpha=3, beta=2.2, gamma=1, mu=0.001):
    dx = alpha*x*(1-y)-beta*z
    dy = -gamma*y*(1-np.power(x,2))
    dz = mu*x
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def burkeShawAttractor(x, y, z, s = 10, v = 4.272):
    dx = -s*(x+y)
    dy = -y-s*x*z
    dz = s*x*y + v
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def yuWangAttractor(x, y, z, a = 10, b = 40, c = 2, d = 2.5):
    dx = a*(y-x)
    dy = b*x-c*x*z
    dz = np.exp(x*y)-d*z
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def chenCelilovskyAttractor(x, y, z, a = 36, b = 3, c = 20):
    dx = a*(y-x)
    dy = c*y - x*z
    dz = x*y - b*z
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def denTsucsAttractor(x, y, z, a = 40, c = 0.833, d = 0.5, e = 0.65, f = 20):
    dx = a*(y-x)+d*x*z
    dy = f*y - x*z
    dz = c*z + x*y - e*np.power(x,2)
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def arneodoAttractor(x, y, z, a = -5.5, b = 3.5, c = -1.0):
    dx = y
    dy = z
    dz = -a*x - b*y - z + c*np.power(x,3)
    return(np.array([dx, dy, dz]))

# dt = 0.00001
# pos_init = np.array([1, 1, 0])

def dequanLiAttractor(x, y, z, alpha = 40, beta = 1.833, delta = 0.16, epsilon = 0.65, rho = 55, zeta = 20):
    dx = alpha*(y-x) + delta*x*z
    dy = rho*x + zeta*y - x*z
    dz = beta*z + x*y - epsilon*np.power(x,2)
    return(np.array([dx, dy, dz]))

# dt = 0.001
# pos_init = np.array([1, 1, 0])

def financeAttractor(x, y, z, alpha = 0.001, beta = 0.2, zeta = 1.1):
    dx = ((1/beta)-alpha)*x + z + x*y
    dy = -beta*y - np.power(x,2)
    dz = -x - zeta*z
    return(np.array([dx, dy, dz]))

# dt = 0.0001
# pos_init = np.array([0.1, 0.1, 0.1])

def genesioTesiAttractor(x, y, z, a1 = 1, a2 = 1.1, a3 = 0.44, a4 = 1):
    dx = y
    dy = z
    dz = -a1*x - a2*y - a3*z + a4*np.power(x,2)
    return(np.array([dx, dy, dz]))

# dt = 0.0001
# pos_init = np.array([0.1, 0.1, 0.1])

def hadleyAttractor(x, y, z, alpha=0.2, beta=4, zeta=8, delta=1):
    dx = -np.power(y,2)-np.power(z,3)-alpha*x+alpha*zeta
    dy = x*y-beta*x*z-y+delta
    dz = beta*x*y+x*z-z
    return(np.array([dx, dy, dz]))

def liuChenAttractor(x, y, z, alpha = 2.4, beta = -3.78, sigma = 14, delta = -11, epsilon = 4, zeta = 5.58, rho = 1):
    dx = alpha*y + beta*x + sigma*y*z
    dy = delta*y - z + epsilon*x*z
    dz = zeta*z+rho*x*y
    return(np.array([dx, dy, dz]))

# dt = 0.00001
# pos_init = np.array([-0.1, 0.5, -0.6])

def chenAttractor(x, y, z, a = 40, b = 3, c = 28):
    dx = a*(y-x)
    dy = (c-a)*x - x*z + c*y
    dz = x*y - b*z
    return(np.array([dx, dy, dz]))

def sprottLinzFAttractor(x, y, z, a = 0.5):
    dx = y + z
    dy = -x + a*y
    dz = np.power(x,2) - z
    return(np.array([dx, dy, dz]))

def sprottBAttraction(x, y, z, a = 0.4, b = 1.2, c =1):
    dx = a*y*z
    dy = x - b*y
    dz = c - x*y
    return(np.array([dx, dy, dz]))