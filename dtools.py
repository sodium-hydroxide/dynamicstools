# Set of functions for dynamical systems written by Noa3h J. Blair
# Imports packages used
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

###############################################################
#                                                             #
#                      General Tools                          #
#                                                             #
###############################################################

###############################################################
#                                                             #
#   Tools for Poincare Plots of periodically driven systems   #
#                                                             #
###############################################################
## Tools for Sinusoidal Forces
# These are for a system with driving force of the form
    # F(t) = g cos(2 pi t)
    # where g is strength of force
    # the period is fixed at one so that the equations are a little nicer




# Function to use runga kutta solve system over one period
# dq/dt = p
# dp/dt = f(q,p) + g*cos(2 pi t)
    # time_ind_force:= the time independent force (q,p); function
    # q:= position; float
    # p:= generalized momentum; float
    # g:= driving strength
def one_period_sol(time_ind_force,qo,po,g):
    # time step for runga-kutta
    dt = 1.0/50.0
    # starts q and p at their initial values
    q = qo
    p = po
    for j in range(0,50):
        # current value of time
        t = dt*j
        # the runga-kutta 45 correction terms
        k1q = dt*p
        k1p = dt*(time_ind_force(q,p) + g*np.cos(2*np.pi*t))
        k2q = dt*(p + (0.5*k1p))
        k2p = dt*(time_ind_force(q + (0.5*k1p),p + (0.5*k1p)) + g*np.cos(2*np.pi*(t+0.5*dt)))
        k3q = dt*(p + (0.5*k2p))
        k3p = dt*(time_ind_force(q + (0.5*k2p),p + (0.5*k2p)) + g*np.cos(2*np.pi*(t+0.5*dt)))
        k4q = dt*(p + k3p)
        k4p = dt*(time_ind_force(q + k3p,p + k3p) + g*np.cos(2*np.pi*(t+dt)))
        q += (1.0/6.0) *(k1q + 2*k2q + 2*k3q + k4q)
        p += (1.0/6.0) *(k1p + 2*k2p + 2*k3p + k4p)
    return q,p

# Generates the first n values of the poincare map
    # time_ind_force:= the time independent force (q,p); function
    # qo:= the initial position; float
    # po:= the initial momentum; float
    # g:= the strength of driving force; float
    # n:= number of periods; int
def poincare(time_ind_force,qo,po,g,n):
    # initializes the arrays
    qvals = np.zeros(n)
    pvals = np.zeros(n)
    qvals[0] = qo
    pvals[0] = po
    # computes the evolution of each point and then adds that point to the array
    for k in range(1,n):
        q,p = one_period_sol(time_ind_force,qvals[k-1],pvals[k-1],g)
        qvals[k] = q
        pvals[k] = p
    return qvals, pvals

# Plots poincare map for damped driven pendulum over n periods
    # time_ind_force:= the time independent force (q,p); function
    # qo:= the initial position; float
    # po:= the initial momentum; float
    # g:= the strength of driving force; float
    # n:= number of periods; int
def poincare_plot(time_ind_force,qo,po,g,n):
    qvals, pvals = poincare(time_ind_force,qo,po,g,n)
    plt.scatter(qvals,pvals)
    plt.show()

## Common unforced systems (called by sinus_solution)
# for all of these
    # q:= current position; float
    # p:= current momentum; float
# Linear oscillator
def linear_osc(q,p):
    # damping coefficient and natural frequency respectively
    b = 0.5
    w = 6.0
    # gives force
    force = -2.0*b*p - (w**2)*q
    return force
# Damped Driven Pendulum
def ddpend(q,p):
    # damping coefficient and natural frequency respectively
    b = 0.5
    w = 6.0
    # gives force
    force = -2.0*b*p - (w**2)*np.sin(q)
    return force
# Duffing Equation
def duffing(q,p):
    force = q - (q**3)
    return force

###############################################################
#                                                             #
#          Tools for periodically impulsed systems            #
#                                                             #
###############################################################

### Set of tools for a periodically impulsed particle
# solves for the first n values of the recursion relation
# q_n+1 = p_n + q_n + P(q_n)
# p_n+1 = p_n + P(q_n)
    # impulse := the impulse as a function of q; function
    # qo:= initial coordinate; float
    # po:= initial momentum; float
    # n:= number of periods to solve for; int
def periodic_impulse(impulse,qo,po,n):
    q = np.zeros(n)
    p = np.zeros(n)
    q[0] = qo
    p[0] = po
    for j in range(1,n):
        q[j] = q[j-1] + p[j-1] + impulse(q[j-1])
        p[j] = p[j-1]
    return q,p

# Plots solutions to periodically impulsed system
    # impulse := the impulse as a function of q; function
    # qo:= initial coordinate; float
    # po:= initial momentum; float
    # n:= number of periods to solve for; int
def periodic_impulse_plot(impulse,qo,po,n):
    q,p = periodic_impulse(impulse,qo,po,n)
    plt.scatter(q,p)
    plt.show()


def the_force(q):
    return -1*q




###############################################################
#                                                             #
#                Tools for stochastic systems                 #
#                                                             #
###############################################################



###############################################################
#                                                             #
#             Tools for Classical Chaotic Maps                #
#                                                             #
###############################################################